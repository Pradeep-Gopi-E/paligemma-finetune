import re
import numpy as np
import torch
import configs.ft_config as object_detection_config
import os

from torch.utils.data import DataLoader
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from datasets import load_dataset

from utils.ft_utils import collate_fn, freeze_layers, extract_objects
from functools import partial
from matplotlib import pyplot as plt, patches

# ---- Disable Torch Inductor ----
# os.environ["TORCH_COMPILE_DISABLE"] = "1"
# os.environ["TORCHDYNAMO_DISABLE"] = "1"
# os.environ["PYTORCH_SDP_KERNEL"] = "math"
# os.environ.pop("TORCH_LOGS", None)
# os.environ["TORCHDYNAMO_VERBOSE"] = "0"

def infer_on_model(model, test_batch, before_pt=True):
    mean = processor.image_processor.image_mean
    std = processor.image_processor.image_std
    

    batch_size = len(test_batch["pixel_values"])
    with torch.inference_mode():
        generated_outputs = model.generate(
            **test_batch, max_new_tokens=100, do_sample=False
        )
        generated_outputs = processor.batch_decode(
            generated_outputs, skip_special_tokens=True
        )

    for index in range(batch_size):
        pixel_value = test_batch["pixel_values"][index].cpu().to(torch.float32)
        unnormalized_image = (pixel_value.numpy() * np.array(std)[:, None, None]) + np.array(mean)[:, None, None]
        unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
        unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)

        element = generated_outputs[index]
        detection_string = element.split("\n")[1]

        if before_pt:
            if detection_string == "":
                print(f"Image {index}: No bbox found")
            else:
                print(f"Image {index}: {detection_string}")
        else:
            objects = extract_objects(detection_string, 448, 448, unique_labels=False)
            plt.figure()
            plt.imshow(unnormalized_image)
            for obj in objects:
                bbox = obj["xyxy"]
                plt.gca().add_patch(
                    patches.Rectangle(
                        (bbox[0], bbox[1]),
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1],
                        linewidth=2,
                        edgecolor="r",
                        facecolor="none"
                    )
                )
                plt.text(bbox[0], bbox[1]-10, obj["name"], color="red", fontsize=12, weight="bold")
            plt.savefig(f"debug_bbox_image_{index}.png")
            plt.close()

if __name__ == "__main__":
    # get the device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # load the dataset
    print(f"[INFO] loading {object_detection_config.DATASET_ID} from hub...")
    train_dataset = load_dataset(object_detection_config.DATASET_ID, split="train")
    test_dataset = load_dataset(object_detection_config.DATASET_ID, split="test")
    print(f"[INFO] {len(train_dataset)=}")
    print(f"[INFO] {len(test_dataset)=}")

    # get the processor
    print(f"[INFO] loading {object_detection_config.MODEL_ID} processor from hub...")
    processor = AutoProcessor.from_pretrained(object_detection_config.MODEL_ID)

    # build the data loaders
    print("[INFO] building the data loaders...")
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=partial(
            collate_fn,
            image_title="image",
            prompt="detect circle; triangle.",
            suffix_title="label_for_paligemma",
            processor=processor,
            device=device,
            train=True,
        ),
        batch_size=object_detection_config.BATCH_SIZE,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=partial(
            collate_fn,
            image_title="image",
            prompt="detect circle; triangle.",
            suffix_title="label_for_paligemma",
            processor=processor,
            device=device,
            train=False,
        ),
        batch_size=object_detection_config.BATCH_SIZE,
        shuffle=False,
    )

    # load the pre trained model
    print(f"[INFO] loading {object_detection_config.MODEL_ID} model...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        object_detection_config.MODEL_ID,
        torch_dtype=object_detection_config.MODEL_DTYPE,
        device_map=device,
        revision=object_detection_config.MODEL_REVISION,
    )

    # freeze the weights
    print(f"[INFO] freezing the model weights...")
    model = freeze_layers(model, not_to_freeze="attn")

    # run model generation before fine tuning
    test_batch = next(iter(test_dataloader))
    infer_on_model(model, test_batch)

    # fine tune the model
    print("[INFO] fine tuning the model...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=object_detection_config.LEARNING_RATE
    )
    for epoch in range(object_detection_config.EPOCHS):
        for idx, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            if idx % 500 == 0:
                print(f"Epoch: {epoch} Iter: {idx} Loss: {loss.item():.4f}")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    print("[INFO] fine tuning complete!")
    # Save the fine-tuned model locally
    save_path = "./paligemma-finetuned"
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"[INFO] Model saved locally at {save_path}")


    # run model generation after fine tuning
    infer_on_model(model, test_batch, before_pt=False)
    
    push = input("Do you want to push the model to Hugging Face Hub? (yes/no): ").strip().lower()
if push == "yes":
    model.push_to_hub("paligemma-finetuned_224")
    processor.push_to_hub("paligemma-finetuned_224")
    print("[INFO] Model pushed to Hugging Face Hub!")
else:
    print("[INFO] Model not pushed to hub.")
