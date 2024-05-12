from os import listdir
from os.path import join, isfile, splitext, getmtime, basename

from termcolor import colored
import click

from srcnn import (
    pick_device,
    load_model,
    save_model_onnx,
    train as srcnn_train,
    save_model,
    upscale as srcnn_upscale,
    save_image,
)

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


def get_device_and_model(new_model: bool, cpu: bool):
    [device, device_name] = pick_device(cpu)
    print(colored("Using device:", "blue"), f"'{device_name}'({device})")

    model_path = find_model_file(new_model)
    model = load_model(model_path)
    model = model.to(device)
    print(colored("Model:", "blue"), model)
    return device, model


def generate_date_for_filename():
    from datetime import datetime

    now = datetime.now()
    return now.strftime("%Y-%m-%d--%H-%M-%S")


@click.command()
@click.option(
    "--input",
    "-i",
    type=click.Path(exists=True),
    help="Directory with the training samples.",
)
# @click.argument("name", type=click.STRING, required=False)
@click.option("--epochs", "-e", default=50, help="How long to train for.")
@click.option(
    "--new-model",
    "-n",
    is_flag=True,
    default=False,
    help="Starts training from scratch instead of continuing last session.",
)
@click.option("--cpu", is_flag=True, default=False, help="Force to use CPU.")
def train(input: str, epochs: int, new_model: bool, cpu: bool):
    device, model = get_device_and_model(new_model, cpu)

    batch_size = 5
    learing_rate = 0.01
    srcnn_train(device, model, input, epochs, batch_size, learing_rate)
    print(colored("Training finshed", "green"))

    # save model params
    out_date = generate_date_for_filename()
    out_filename = join(MODELS_DIR, f"srcnn_model.{out_date}.pt")
    print(colored("Saving model parameters to:", "blue"), f"'{out_filename}'")
    save_model(model, out_filename)


@click.command()
@click.option(
    "--input",
    "-i",
    type=click.Path(exists=True),
    help="Image to upscale.",
)
@click.option("--cpu", is_flag=True, default=False, help="Force to use CPU.")
def upscale(input: str, cpu: bool):
    device, model = get_device_and_model(False, cpu)

    result = srcnn_upscale(device, model, input)
    print(colored("Upscale finshed", "green"))

    # save result
    base = basename(input)
    filename = splitext(base)[0]
    out_date = generate_date_for_filename()
    out_filename = join(OUTPUT_DIR, f"{filename}.{out_date}.png")
    print(colored("Saving result to:", "blue"), f"'{out_filename}'")
    save_image(result, out_filename)


@click.command()
@click.option(
    "--output", "-o", type=click.Path(), help="Result filepath.", default="srcnn.onnx"
)
def export_onnx(output):
    """Export the model to ONNX file"""
    device, model = get_device_and_model(new_model=False, cpu=True)
    print(colored("Saving result to:", "blue"), f"'{output}'")
    save_model_onnx(model, output)


@click.group()
def main():
    f"""SRCNN app. By default, latest model from './{MODELS_DIR}' is used."""


if __name__ == "__main__":
    main.add_command(train)
    main.add_command(upscale)
    main.add_command(export_onnx)
    main()
