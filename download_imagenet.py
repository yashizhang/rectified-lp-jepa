import os
from datasets import load_dataset
from PIL import Image

# Output directory
out_dir = "./data/imagenet100"
os.makedirs(out_dir, exist_ok=True)

# Load dataset (one widely used variant)
dataset = load_dataset("clane9/imagenet-100")

def save_split(split_name):
    split = dataset[split_name]
    for i, sample in enumerate(split):
        label = sample["label"]
        img: Image.Image = sample["image"]

        # Create label folder
        label_dir = os.path.join(out_dir, split_name, str(label))
        os.makedirs(label_dir, exist_ok=True)

        # Save image
        img.save(os.path.join(label_dir, f"{i}.jpg"))

# Save both splits
save_split("train")
save_split("validation")

print("Done.")