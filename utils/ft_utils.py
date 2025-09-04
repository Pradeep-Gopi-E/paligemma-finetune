import torch
import re
from matplotlib import pyplot as plt, patches
import numpy as np

DETECT_RE = re.compile(
    r"(.*?)" + r"((?:<loc\d{4}>){4})\s*" + r"([^;<>]+) ?(?:; )?",
)


def extract_objects(detection_string, image_width, image_height, unique_labels=False):
    objects = []
    seen_labels = set()

    while detection_string:
        match = DETECT_RE.match(detection_string)
        if not match:
            break

        prefix, locations, label = match.groups()
        location_values = [int(loc) for loc in re.findall(r"\d{4}", locations)]
        y1, x1, y2, x2 = [value / 1024 for value in location_values]
        y1, x1, y2, x2 = map(
            round,
            (y1 * image_height, x1 * image_width, y2 * image_height, x2 * image_width),
        )

        label = label.strip()  # Remove trailing spaces from label

        if unique_labels and label in seen_labels:
            label = (label or "") + "'"
        seen_labels.add(label)

        objects.append(dict(xyxy=(x1, y1, x2, y2), name=label))

        detection_string = detection_string[len(match.group()) :]

    return objects


def collate_fn(examples, image_title, prompt, suffix_title, processor, device, train):
    images = [example[image_title].convert("RGB") for example in examples]

    prompt = [prompt for _ in examples]
    if train:
        suffix = [example[suffix_title] for example in examples]
    else:
        suffix = None

    inputs = processor(
        images=images,
        text=prompt,
        suffix=suffix,
        return_tensors="pt",
        padding="longest",
    )

    inputs = inputs.to(torch.bfloat16).to(device)
    return inputs

def freeze_layers(model, not_to_freeze):
    for name, param in model.named_parameters():
        if not_to_freeze in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model

