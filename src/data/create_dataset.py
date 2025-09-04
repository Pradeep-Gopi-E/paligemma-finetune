import os
from PIL import Image
import yaml
from datasets import Dataset, DatasetDict, Image as ImageFeature  
from tqdm import tqdm
from configs.dataset_config import (
    DATASET_ROOT, DATA_YAML, OUTPUT_ROOT, IMAGE_FORMATS,
    SPLITS, OCCLUSION_TYPES, CONTRAST_TYPES,
    PUSH_TO_HUB, HUB_REPO_PREFIX
)


with open(DATA_YAML, "r") as f:
    data_cfg = yaml.safe_load(f)

class_names = data_cfg["names"]  


def yolo_to_xyxy(box, img_width, img_height):
    x_center, y_center, w, h = box
    x_center *= img_width
    y_center *= img_height
    w *= img_width
    h *= img_height
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
    return [x1, y1, x2, y2]

def format_location(value, max_value):
    return f"<loc{int(round(value * 1024 / max_value)):04}>"

def convert_to_paligemma_string(boxes, classes, img_width, img_height):
    detection_strings = []
    for bbox, cls_id in zip(boxes, classes):
        x1, y1, x2, y2 = bbox
        locs = [
            format_location(y1, img_height),
            format_location(x1, img_width),
            format_location(y2, img_height),
            format_location(x2, img_width),
        ]
        name = class_names[int(cls_id)]
        detection_strings.append("".join(locs) + f" {name}")
    return " ; ".join(detection_strings)

def find_image_path(label_path):
    for ext in IMAGE_FORMATS:
        img_path = label_path.replace("labels", "images").replace(".txt", ext)
        if os.path.exists(img_path):
            return img_path
    return None

def parse_split(split_path):
    data = []
    for root, _, files in os.walk(split_path):
        if root.endswith("labels"):
            for file in files:
                if file.endswith(".txt"):
                    label_path = os.path.join(root, file)
                    img_path = find_image_path(label_path)
                    if not img_path:
                        continue

                    img = Image.open(img_path)
                    img_width, img_height = img.size

                    with open(label_path, "r") as lf:
                        lines = lf.readlines()

                    boxes = []
                    classes = []
                    for line in lines:
                        parts = line.strip().split()
                        cls_id = int(parts[0])
                        x_center, y_center, w, h = map(float, parts[1:])
                        boxes.append(yolo_to_xyxy([x_center, y_center, w, h], img_width, img_height))
                        classes.append(cls_id)

                    paligemma_label = convert_to_paligemma_string(boxes, classes, img_width, img_height)

                    data.append({
                        "image": img_path,
                        "label_for_paligemma": paligemma_label
                    })
    return data


for occ in OCCLUSION_TYPES:
    for contrast in CONTRAST_TYPES:
        print(f"\n[INFO] Processing: {occ} -> {contrast}")

        dataset_dict = {}
        for split in SPLITS:
            split_path = os.path.join(DATASET_ROOT, split, occ, contrast)
            if not os.path.exists(split_path):
                print(f"[WARN] Split path not found: {split_path}")
                continue

            split_data = parse_split(split_path)
            dataset_dict[split if split != "val" else "validation"] = (
        Dataset.from_list(split_data).cast_column("image", ImageFeature()) # Changed `Image()` to `ImageFeature()`
    )

        if dataset_dict:
            dataset = DatasetDict(dataset_dict)

            # Save locally
            save_path = os.path.join(OUTPUT_ROOT, f"{occ}_{contrast}_paligemma")
            print(f"[INFO] Saving to: {save_path}")
            os.makedirs(save_path, exist_ok=True)
            dataset.save_to_disk(save_path)

            # Push to Hugging Face Hub (optional)
            if PUSH_TO_HUB:
                repo_name = f"{HUB_REPO_PREFIX}/{occ}-{contrast}-paligemma"
                print(f"[INFO] Pushing to Hub: {repo_name}")
                dataset.push_to_hub(repo_name)
                print(f"[INFO] Pushed to Hub: {repo_name}")

print("\n[INFO] All processing completed!")