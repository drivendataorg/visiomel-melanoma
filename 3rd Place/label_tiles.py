"""
Label a set of images.
You can stop and resume labelling any time. Labels are saved regularly.
"""
import argparse
import os
import tkinter as tk

import pandas as pd
from PIL import Image, ImageTk

# Set up the argument parser
parser = argparse.ArgumentParser(description="Label images.")
parser.add_argument(
    "-i", "--image_dir", type=str, required=True, help="The directory containing your images"
)
parser.add_argument("-c1", "--class_1", type=str, required=True, help="The name of the first class")
parser.add_argument(
    "-c2", "--class_2", type=str, required=True, help="The name of the second class"
)
parser.add_argument(
    "-o",
    "--output_file",
    type=str,
    default="labels.csv",
    help="The name of the output file, containing labels for each image.",
)

args = parser.parse_args()
image_dir = args.image_dir
class_1 = args.class_1
class_2 = args.class_2
key_1 = "Left"
key_2 = "Right"
output_file = args.output_file
checkpoint_interval = 10
labeled_images = []
labeled_image_filenames = set()
if os.path.exists(output_file):
    df = pd.read_csv(output_file)
    labeled_images = df.to_dict("records")
    labeled_image_filenames = set(df["filename"])

image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".png")]
counter = 0
image_index = 0
root = tk.Tk()
root.eval("tk::PlaceWindow . center")
instructions = tk.Label(
    root, text=f"Press '{key_1}' to label as {class_1}, '{key_2}' to label as {class_2}"
)
instructions.pack()

image_label = tk.Label(root)
image_label.pack()


def show_next_image():
    global image_index
    global instructions
    instructions.config(text=f"n_image {counter}")

    while image_index < len(image_files):
        image_file = image_files[image_index]
        if image_file not in labeled_image_filenames:
            break
        image_index += 1

    if image_index >= len(image_files):
        print("Finished labeling all images")
        root.quit()
        return

    image = Image.open(os.path.join(image_dir, image_file))
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo


def label_image(label):
    global counter, image_index
    image_file = image_files[image_index]
    labeled_images.append({"filename": image_file, "label": label})
    labeled_image_filenames.add(image_file)
    print(f"Labeled {image_file} as {label}")
    counter += 1
    image_index += 1
    if counter % checkpoint_interval == 0:
        df = pd.DataFrame(labeled_images)
        df.to_csv(output_file, index=False)
        print(f"Saved checkpoint after labeling {counter} images")
    show_next_image()


def on_key_press(event):
    if event.keysym == key_1:
        label_image(class_1)
    elif event.keysym == key_2:
        label_image(class_2)


root.bind("<KeyPress>", on_key_press)
show_next_image()
root.mainloop()
df = pd.DataFrame(labeled_images)
df.to_csv(output_file, index=False)
