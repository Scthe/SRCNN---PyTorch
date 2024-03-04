# import inject_external_torch_into_path
# inject_external_torch_into_path.inject_path()

import argparse
from os import listdir
from os.path import join, isfile, splitext, getmtime, basename

from termcolor import colored

from srcnn import pick_device, load_model, train, save_model, upscale, save_image

"""
TODO
- Upscale rbg color too? Not just luma? -rgb flag
"""

MODELS_DIR = "models"
OUTPUT_DIR = "images_upscaled"
MODEL_EXT = ".pt"


def find_model_file(force_new):
    def is_model_file(f):
        fullpath = join(MODELS_DIR, f)
        # print(fullpath, isfile(fullpath), splitext(f)[1])
        return isfile(fullpath) and splitext(f)[1] == MODEL_EXT

    if force_new:
        return None

    models = [join(MODELS_DIR, f) for f in listdir(MODELS_DIR) if is_model_file(f)]
    # print(models)
    if not models:
        print("No previous model found")
        return None

    models.sort(key=lambda x: getmtime(x))
    return models[-1]


def generate_date_for_filename():
    from datetime import datetime

    now = datetime.now()
    return now.strftime("%Y-%m-%d--%H-%M-%S")


def args_require(args, key):
    value = vars(args)[key]
    if value is None:
        raise Exception(f"Missing program arg: '{key}'")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"SRCNN app. By default, latest model from './{MODELS_DIR}' is used",
    )

    # https://docs.python.org/3/library/argparse.html
    # https://stackoverflow.com/questions/20063/whats-the-best-way-to-parse-command-line-arguments
    parser.add_argument("--train", "-t", action="store_true", help="Training mode")
    parser.add_argument(
        "--epochs", "-e", type=int, default=50, help="[Training] How long to train for"
    )
    parser.add_argument(
        "--new-model",
        "-n",
        action="store_true",
        help="[Training] Starts training from scratch",
    )
    parser.add_argument("--cpu", action="store_true", help="Force to use CPU")
    parser.add_argument(
        "--input",
        "-i",
        help=f"[Upscale] Image to upscale. [Training] Directory with training samples",
    )

    args = parser.parse_args()
    # print(args)

    [device, device_name] = pick_device(args.cpu)
    print(colored("Using device:", "blue"), f"'{device_name}'({device})")

    model_path = find_model_file(args.new_model)
    model = load_model(model_path)
    model = model.to(device)
    print(colored("Model:", "blue"), model)

    if args.train:
        samples_dir = args_require(args, "input")

        batch_size = 5
        learing_rate = 0.01
        train(device, model, samples_dir, args.epochs, batch_size, learing_rate)
        print(colored("Training finshed", "green"))

        # save model params
        out_date = generate_date_for_filename()
        out_filename = join(MODELS_DIR, f"srcnn_model.{out_date}.pt")
        print(colored("Saving model parameters to:", "blue"), f"'{out_filename}'")
        save_model(model, out_filename)
    else:
        img_path = args_require(args, "input")
        result = upscale(device, model, img_path)
        print(colored("Upscale finshed", "green"))

        # save result
        base = basename(img_path)
        basename = splitext(base)[0]
        out_date = generate_date_for_filename()
        out_filename = join(OUTPUT_DIR, f"{basename}.{out_date}.png")
        print(colored("Saving result to:", "blue"), f"'{out_filename}'")
        save_image(result, out_filename)

    print("--- DONE ---")
