# SRCNN - PyTorch ([DEMO](https://scthe.github.io/SRCNN-PyTorch/))

Currently, there are 2 predominant upscalers on [Civitai](https://civitai.com/): [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/tree/master) and [UltraSharp](https://openmodeldb.info/models/4x-UltraSharp). Both are based on [ESRGAN](https://arxiv.org/pdf/1809.00219.pdf). If you look at any recent paper regarding Super-Resolution, you will see sentences like:

> "Since the pioneering work of SRCNN [9], deep convolution neural network (CNN) approaches have brought prosperous developments in the SR field"
>
> -- <cite>"Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data" by Wang et. all.</cite>

SRCNN? This sounds familiar. In 2015 I [wrote](https://github.com/Scthe/cnn-Super-Resolution) an implementation in raw OpenCL that runs on GPU. This repo is a PyTorch reimplementation that I wrote some time ago. I also had a TensorFlow one but seems to be lost in the depths of the hard drive.

## Overview

Super-resolution problem tries to upscale the image so that perceived loss of quality is minimal. For example, after scaling with bicubic interpolation it is apparent that some pixels are just smudged together. The question is: can AI do a better job?

## Results

![gh_image_compare](https://github.com/Scthe/SRCNN---PyTorch/assets/9325337/2b526188-220c-4dd8-b648-cdacd125a449)

_left: upscaling with bicubic interpolation, right: result of the presented algorithm_

![gh_image_details](https://github.com/Scthe/SRCNN---PyTorch/assets/9325337/43eec73c-f814-472b-992e-ba9adda9cb53)

_Details closeup - left: upscaling with bicubic interpolation, right: result of the presented algorithm_

The current algorithm only upscales the luma, the chroma is preserved as-is. This is a common trick known as [chroma subsampling](https://en.wikipedia.org/wiki/Chroma_subsampling).

## Usage

### Install dependencies

`pip install -r requirements.txt` will install the CPU version of PyTorch. If you want to run the code on GPU, use `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118` ([docs](https://pytorch.org/get-started/locally/)). Be wary that it's a **much** bigger download size and you don't need it - the model is small enough.

Alternatively, you can reuse packages from your [kohya_ss](https://github.com/bmaltais/kohya_ss) or [Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui). Add it to Python's path:

```python
def inject_path():
    import sys
    sys.path.append('C:/programs/install/kohya_ss/venv/Lib/site-packages') # your path here
inject_path() # call this fn from both 'main.py' and 'gen_samples.py'
```

### Training

1. Put some images into `./images_raw`.
2. `python gen_samples.py 400 -64`. Generate 400 image pairs (one image was downscaled, the other is the original size). Each image is 64x64 px and it's stored in `./samples_gen_64`.
3. `python main.py train --cpu -n -e 20 -i "samples_gen_64"`. Run training for 20 epochs using samples from `./samples_gen_64`. By default:
   - The program will use GPU if appropriate PyTorch is installed. Use `--cpu` flag to force to use the CPU (even if you have GPU-capable PyTorch).
   - The program will continue from the last checkpoint (stored in `./models`). Use `-n` to start from scratch.

First, we need to generate training data. `./gen_samples.py` reads images from `./images_raw` and randomly crops 32x32 px (or 64x64 px with `-64`) patches. They will be stored as e.g. `./samples_gen_64/0b0mkhrd.large.png`. We also generate corresponding `./samples_gen_64/0b0mkhrd.small.png`. It's done by downscaling and upscaling the cropped image. Our goal is to learn how to turn the blurred small image into the sharp one.

If you want to get something good enough, the training will take a few minutes at most, even on the CPU.

After training, the model is saved to e.g. `./models/srcnn_model.2024-02-27--23-43-05.pt`

### Inference

- `python main.py upscale -i "<some path here>/image_to_upscale.jpg"`. Run `main.py` with `-i` set to your image.

The program will automatically separate luma, run upscale, and reassemble the final image. The `--cpu` flag works here too. By default, it will use the latest model from the `./models` directory.

The result is stored to e.g. `'./images_upscaled/<your_image_name>.2024-02-27--23-43-27.png'`.


## Web demo

The PyTorch model was exported to [ONNX file](https://onnxruntime.ai/docs/tutorials/web/classify-images-nextjs-github-template.html). This allows inference in the web browser.  Unfortunately, ONNX runtime on the web has errors that prevent using GPU backends (WebGPU, WebGL). CPU is much slower. Fortunately, this app is just my private playground. Use [netron.app](https://netron.app/) to preview the [srcnn.onnx](web/srcnn.onnx) file.

### Lessons from ONNX conversion

1. During the training, your image-based PyTorch model has input of size `[batch_size, img_channel_count, img_height, img_width]`. During inference, Pytorch accepts e.g. `[img_channel_count, img_height, img_width]`. It does not mind that the dimension for `batch_size` does not exist. **THIS IS NOT TRUE FOR ONNX!**.
2. Double check you have always correct tensors for images: `[batch_size, img_channel_count, img_height, img_width]`. I've lost "a bit" of time cause my input had width and height reversed. Evident when:
    - Model works only for square images.
    - Vertical images have a few "ghost duplicates" along horizontal axis.
    - Horizontal images have many "ghost duplicates" along horizontal axis.

The second one sounds silly. But after years of writing code for CG, your fingers do not think about it.

I recommend following utils (for single grayscale image processing):

```js
const encodeDims = (w, h) => [1, 1, h, w]; // [batch_size, channels, height, width]
const decodeDims = (dims) => [dims[3], dims[2]]; // returns: [w, h]
```

## The files

- `images_raw/`. The original images we will use to generate training samples from. Add some images to this directory.
- `images_upscaled/`. Contains the final upscaled image after inference.
- `models/`. Contains learned models as `.pt` file.
- `samples_gen_32/`. Training patches generated from `images_raw` with `gen_samples.py` with default patch size (32x32 px).
- `samples_gen_64/`. Training patches generated from `images_raw` with `gen_samples.py` with `-64` flag (64x64 px).
- `gen_samples.py/`. Script to generate sample patches from `images_raw`.
- `main.py`. CLI for training/inference.
- `srcnn.py`. CNN model implementation.

## References

If you are interested in math or implementation details, I've written 2 articles 9 years ago:

- ["Math behind (convolutional) neural networks"](https://www.sctheblog.com/blog/math-behind-neural-networks/)
- ["Writing neural networks from scratch - implementation tips"](https://www.sctheblog.com/blog/neural-networks-implementation-tips/)

Ofc. the original ["Image Super-Resolution Using Deep Convolutional Networks"](https://arxiv.org/abs/1501.00092) is still relevant. Even the current state of the art references it as the progenitor.
