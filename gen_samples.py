# import inject_external_torch_into_path
# inject_external_torch_into_path.inject_path()

import argparse
from random import sample 
import os
from os.path import join, splitext

from termcolor import colored
from torchvision.utils import save_image as torch_save_image
from torchvision.io import read_image

from main import generate_date_for_filename

RAW_IMAGES_DIR = 'images_raw'
OUTPUT_DIR = 'samples_gen'

'''
- https://pytorch.org/vision/stable/transforms.html#v2-api-reference-recommended
'''

def list_input_images(dir_):
    image_paths = [join(dir_, f) for f in os.listdir(dir_)]
    result = []
    allowed_ext = ['.jpg', '.jpeg', '.png']
    for img_path in image_paths:
        ext = splitext(img_path)[1]
        if ext not in allowed_ext:
            if not img_path.endswith('.gitkeep'):
                print(colored(f'Invalid file found in {dir_}', 'red'), img_path)
        else:
            result.append(img_path)
    return result

def random_string(size):
    import random
    import string
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=size))

def generate_sample(img, sample_size):
    import torch
    from torchvision.transforms import v2

    transforms = v2.Compose([
        v2.RandomCrop(sample_size),
        v2.ToDtype(torch.float32, scale=True),
    ])
    img_large = transforms(img)
    transforms = v2.Compose([
        # blur
        v2.Resize(sample_size // 2, antialias=True),
        v2.Resize(sample_size, antialias=True),
    ])
    img_small = transforms(img_large)
    return [img_large, img_small]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=f'Generate samples for SRCNN. By default it will be 32x32 px'
    )
    parser.add_argument('count',
        type=int,
        help='Samples count')
    parser.add_argument('--gen-64', '-64',
        action='store_true',
        help='Generate samples of size 64x64 px instead of 32x32 px'
    )

    args = parser.parse_args()
    # print(args)
    sample_size = 64 if args.gen_64 else 32
    count = args.count
    if count <= 0:
        raise Exception(f"Count has to be > 0. Received: {count}")

    print(colored(f'Will generate {count} samples from images in:', 'blue'), f"'{RAW_IMAGES_DIR}'")

    image_paths = list_input_images(RAW_IMAGES_DIR)
    # image_paths = [image_paths[0]]
    if len(image_paths) == 0:
        raise Exception(f"Found no images in '{RAW_IMAGES_DIR}'")
    # print(image_paths)
   
    output_dir = f'{OUTPUT_DIR}_{sample_size}'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    print(colored(f'Generating samples from {len(image_paths)} images. Store result in: ', 'blue'), f"'{output_dir}'")
    samples = [read_image(img_path) for img_path in image_paths]

    for _ in range(count):
        input_img = sample(samples, 1)
        img_large, img_small = generate_sample(input_img, sample_size)
        file_name = random_string(8)
        out_filename_large = join(output_dir, f'{file_name}.large.png')
        out_filename_small = join(output_dir, f'{file_name}.small.png')
        torch_save_image(img_large, out_filename_large)
        torch_save_image(img_small, out_filename_small)

    print('--- DONE ---')