import os
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image

def detect_with_paligemma(
    model_id,
    image_folder,
    prompt="detect circle",
    device=None,
    dtype=None,
    max_new_tokens=200
):
    """
    Runs PaliGemma model on all images in a folder and returns predictions.

    Args:
        model_id (str): HuggingFace model ID for PaliGemma.
        image_folder (str): Path to the folder containing images.
        prompt (str): Text prompt for the model (default: "detect circle").
        device (str): "cuda" or "cpu" (auto-detect if None).
        dtype (torch.dtype): Data type for inference (auto-detect if None).
        max_new_tokens (int): Maximum tokens for generation.

    Returns:
        list[dict]: Each dict has {'file': filename, 'prediction': text}.
    """

    # ---- Environment Tweaks for Safety ----
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["PYTORCH_SDP_KERNEL"] = "math"
    os.environ.pop("TORCH_LOGS", None)
    os.environ["TORCHDYNAMO_VERBOSE"] = "0"

    # ---- Device and dtype ----
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.float16 if device == "cuda" else torch.float32

    # ---- Load Model & Processor ----
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)

    # ---- Helper: Open image safely ----
    def load_rgb(path):
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    # ---- Collect files ----
    valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_ext)]
    files.sort()

    results = []

    # ---- Process images ----
    with torch.no_grad():
        for fn in files:
            p = os.path.join(image_folder, fn)
            try:
                img = load_rgb(p)
                inputs = processor(text=prompt, images=[img], return_tensors="pt").to(device, dtype=dtype)

                output = model.generate(**inputs, max_new_tokens=max_new_tokens)
                input_len = inputs["input_ids"].shape[-1]
                text = processor.decode(output[0][input_len:], skip_special_tokens=True)

                results.append({"file": fn, "prediction": text})
                print(f"{fn} -> {text}")

            except Exception as e:
                print(f"[ERROR] {fn}: {e}")
                results.append({"file": fn, "error": str(e)})

    return results
