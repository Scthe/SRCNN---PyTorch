import os
import timeit

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image as torch_save_image
from torchvision.io import read_image, ImageReadMode
from termcolor import colored

# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# https://www.kdnuggets.com/building-a-convolutional-neural-network-with-pytorch
# https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/


def pick_device(force_cpu=False):
    device = "cpu"
    name = "CPU"
    if torch.cuda.is_available() and not force_cpu:
        print("CUDA devices:")
        for dev_id in range(torch.cuda.device_count()):
            print(f"\t[{dev_id}]", torch.cuda.get_device_name(dev_id))
        dev_id = 0
        torch.cuda.device(dev_id)
        # print(f"Using device [{dev_id}]", torch.cuda.get_device_name())
        device = f"cuda:{dev_id}"
        name = torch.cuda.get_device_name()
    return [device, name]


class Net(nn.Module):
    def __init__(self, l1_out, l1_conv_size, l2_out, l2_conv_size, l3_conv_size):
        super(Net, self).__init__()

        conv_kwargs = {
            "stride": 1,
            "padding": "same",
            # "padding_mode": "reflect",
            "padding_mode": "zeros",
            "bias": True,
        }
        self.conv1 = nn.Conv2d(1, l1_out, l1_conv_size, **conv_kwargs)
        self.conv2 = nn.Conv2d(l1_out, l2_out, l2_conv_size, **conv_kwargs)
        self.conv3 = nn.Conv2d(l2_out, 1, l3_conv_size, **conv_kwargs)

    def forward(self, x):
        # input 64x64x1, output 64x64xl1_out
        x = F.relu(self.conv1(x))
        # input 64x64x64x64xl1_out, output 64x64xl2_out
        x = F.relu(self.conv2(x))
        # input 64x64xl2_out, output 64x64x1
        x = self.conv3(x)
        # x = self.conv2(x)
        return x

    @property
    def device(self):
        ps = self.parameters()
        return ps.__next__().device


def load_model(filepath, force_new_model=False):
    if filepath == None or force_new_model:
        print(colored("Creating new model", "blue"))
        model = Net(64, 9, 32, 1, 5)
    elif os.path.isfile(filepath):
        print(colored("Loading model from:", "blue"), f"'{filepath}'")
        model = torch.load(filepath)
        model.eval()
    else:
        raise Exception(f"Model file not found: '{filepath}'")
    return model


def save_model(model, filepath):
    torch.save(model, filepath)


def save_model_onnx(model, filepath):
    dummy_input = torch.randn(1, 1, 1000, 1000).to(model.device)
    name_in = "upscaled_greyscale_image_IN"
    name_out = "upscaled_greyscale_image_OUT"
    dynamic_axes = {}
    dynamic_axes[name_in] = {
        0: "in_batch_size",
        1: "in_c",
        2: "in_h",
        3: "in_w",
    }
    dynamic_axes[name_out] = {
        0: "out_batch_size",
        1: "out_c",
        2: "out_h",
        3: "out_w",
    }
    print(dynamic_axes)

    torch.onnx.export(
        model,
        dummy_input,
        filepath,
        # verbose=True,
        input_names=[name_in],
        output_names=[name_out],
        # export_params= #Set this to False if you want to export an untrained model
        dynamic_axes=dynamic_axes,
    )


def prepare_image(device, img):
    from torchvision.transforms import v2

    if isinstance(img, str):
        img = read_image(img, mode=ImageReadMode.RGB)
    transforms = v2.Compose(
        [
            v2.Grayscale(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    img = transforms(img)
    img = img.to(device)
    return img


####################################
# TRAINING


class SrcnnDataset(Dataset):
    def __init__(self, device, image_paths):
        self.device = device
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        x_path, y_path = self.image_paths[idx]
        x_image = prepare_image(self.device, x_path)
        y_image = prepare_image(self.device, y_path)
        return x_image, y_image


def list_samples(samples_dir):
    from os import listdir
    from os.path import isfile, join, splitext
    import re

    def is_valid_sample_file(f):
        ext = splitext(f)[1]
        f = join(samples_dir, f)
        return isfile(f) and ext == ".png"

    sample_files = [f for f in listdir(samples_dir) if is_valid_sample_file(f)]
    # print(sample_files)
    samples = {}
    for sample_file in sample_files:
        # parse name for `id` and `size`
        x = re.search(r"([a-z0-9]+?)\.(large|small)\.png", sample_file)
        if x is None:
            raise Exception(f"Invalid sample file name: '{sample_file}'")
        [id, size] = x.groups()
        # print(f"'{sample_file}' id='{id}' size='{size}'")

        image_path = join(samples_dir, sample_file)
        item = samples.get(id, [None, None])
        if size == "small":
            item[0] = image_path
        else:
            item[1] = image_path
        samples[id] = item
    allSamples = list(samples.values())
    result = []
    for x, y in allSamples:
        if x is None or y is None:
            print(f"Invalid large/small image pair: ('{x}','{y}')")
        else:
            result.append([x, y])
    return result


def create_data_loader(device, img_paths, batch_size, is_train=False):
    ds = SrcnnDataset(device, img_paths)
    params = {
        "batch_size": batch_size,
        "shuffle": is_train,
        # 'drop_last': is_train,
        # 'num_workers': 0
    }
    return DataLoader(ds, **params)


def lr_mod_from_batch_size(batch_size):
    """
    # Explanation:

    * If `batch_size==1`, then you update weights once per image. So if you process 10 images,
        it will trigger 10 changes.
    * If `batch_size==10`, then you update weights once per 10 images. So if you process 10 images,
        it will trigger 1 change. It would learn much slower than `batch_size==1`.

    # Solution

    Increase learning rate based on the `batch_size`. Usually it's linear or sqrt.

    @see https://stackoverflow.com/questions/53033556/how-should-the-learning-rate-change-as-the-batch-size-change
    """
    return batch_size


def train(device, model, samples_dir, n_epochs, batch_size, learing_rate):
    if n_epochs <= 0:
        raise Exception(f"Invalid epoch count: {n_epochs}")

    print(colored(f"Reading samples from:", "blue"), f"'{samples_dir}'")
    all_sample_paths = list_samples(samples_dir)
    if len(all_sample_paths) == 0:
        raise Exception(f"No training data found in: '{samples_dir}'")
    print(colored(f"Found {len(all_sample_paths)} samples", "blue"))

    n1 = int(0.9 * len(all_sample_paths))
    train_image_paths = all_sample_paths[:n1]
    train_loader = create_data_loader(
        device, train_image_paths, batch_size, is_train=True
    )
    validation_image_paths = all_sample_paths[n1:]
    validation_loader = create_data_loader(device, validation_image_paths, batch_size)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(
        model.parameters(), lr=learing_rate * lr_mod_from_batch_size(batch_size)
    )
    losses = []
    print(
        colored(f"Starting training", "blue"),
        f"(epochs: {n_epochs}, batch_size: {batch_size}, learing_rate: {learing_rate})",
    )

    for epoch in range(n_epochs):
        start = timeit.default_timer()

        # train
        for in_image, expected in train_loader:
            # Forward pass
            output = model(in_image)
            loss = loss_fn(output, expected)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # evaluate
        loss_acc = 0
        with torch.no_grad():
            for in_image, expected in validation_loader:
                output = model(in_image)
                loss = loss_fn(output, expected)
                # print(in_image.shape, output.shape, loss, loss.item())
                loss_acc += loss.item()

        duration = timeit.default_timer() - start
        loss_avg = loss_acc / len(validation_image_paths)
        losses += [loss_avg]
        max_loss = max(losses)
        iters_left = n_epochs - epoch - 1
        eta = "" if iters_left == 0 else f"ETA {iters_left*duration:4.1f}s"
        progress = (epoch + 1) * 100 // n_epochs
        print(
            colored(f"[{progress:3d}%] Epoch {epoch + 1:5d}:", "magenta"),
            f"Loss: {loss_avg:2.7f} ({loss_avg * 100 / max_loss:5.1f}%). Took {duration:4.2f}s. {eta}",
        )


####################################
# UPSCALE


def save_image(image, filepath):
    torch_save_image(image, filepath)


def replace_luma(img, new_luma):
    """
    Torch greyscale seems to use BT.601. (Rec. 601). Not sure why it's not
    Rec. 709, as like literary everyone else? Seems they just copied from
    TensorFlow. They also do not do gamma correction? As in, they work
    in non-linear space? What is going on?
    Sure, I can gamma correct myself. But for some reason no one does it?
    I mean, neural net does not really care.

    Anyway, here are conversion tables:
    - https://poynton.ca/notes/colour_and_gamma/ColorFAQ.html#RTFToC28
    - https://fourcc.org/fccyvrgb.php - by fourcc!
    - https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion


    # Torch greyscale
    - https://github.com/pytorch/vision/blob/b1123cfd543d35d282de9bb28067e48ebec18afe/torchvision/transforms/_functional_tensor.py#L146
    - luma = (0.2989 * r + 0.587 * g + 0.114 * b)
    """
    print(colored("Replacing luma..", "blue"))
    img = img.to("cpu")
    r, g, b = img.unbind(dim=0)
    pr = (0.500 * r - 0.419 * g - 0.081 * b).to(img.dtype)
    pb = (-0.169 * r - 0.331 * g + 0.500 * b).to(img.dtype)

    w, h = img.shape[1], img.shape[2]
    new_luma = new_luma.reshape(w, h)
    result = torch.empty(3, w, h)
    result[0] = new_luma + 0.000 * pb + 1.402 * pr
    result[1] = new_luma - 0.344 * pb - 0.714 * pr
    result[2] = new_luma + 1.772 * pb + 0.000 * pr
    return result


def upscale(device, model, input_image_path):
    from torchvision.transforms import v2

    print(colored("Upscaling image:", "blue"), f"'{input_image_path}'")
    org_img = read_image(input_image_path, mode=ImageReadMode.RGB)
    org_w, org_h = org_img.shape[1], org_img.shape[2]
    transforms = v2.Compose(
        [
            v2.Resize(size=(org_w * 2, org_h * 2), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    upscaled_img = transforms(org_img)
    print(
        colored("Upscaling from", "blue"),
        org_img.shape,
        colored("to", "blue"),
        upscaled_img.shape,
    )
    my_image = prepare_image(device, upscaled_img)

    with torch.no_grad():
        result_luma = model(my_image)
        return replace_luma(upscaled_img, result_luma)
