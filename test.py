import os

# --- Disable all compiling paths BEFORE importing torch ---
os.environ["TORCH_COMPILE_DISABLE"] = "1"     # disable torch.compile
os.environ["TORCHDYNAMO_DISABLE"] = "1"       # disable TorchDynamo entirely
os.environ["PYTORCH_SDP_KERNEL"] = "math"     # force math attention (no Triton/Flash)
# Optional: quieter logs
os.environ.pop("TORCH_LOGS", None)
os.environ["TORCHDYNAMO_VERBOSE"] = "0"

import torch
# Fallback to eager if anything tries to compile anyway
torch._dynamo.config.suppress_errors = True

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image

# ------------------ Config ------------------
model_id = "google/paligemma2-3b-ft-docci-448"
image_folder = r"C:\Users\prade\OneDrive\Documents\Manav\Pradeep\Synth-RT-DETR-DATASET\test\partial_occlusion\low_contrast\temp"
prompt = "detect circle"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32  # fp16 on GPU helps and avoids some kernels
# --------------------------------------------

# Load model/processor
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(device)
model.eval()
processor = AutoProcessor.from_pretrained(model_id)



# Helper: safe open/convert
def load_rgb(path):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

# Process folder one-by-one (no batching = fewer surprises)
valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_ext)]
files.sort()

with torch.no_grad():
    for fn in files:
        p = os.path.join(image_folder, fn)
        try:
            img = load_rgb(p)
            # Processor already handles resizing to model’s expected size (448), so no manual resize needed
            inputs = processor(text=prompt, images=[img], return_tensors="pt").to(device, dtype=dtype)

            output = model.generate(**inputs, max_new_tokens=200)
            input_len = inputs["input_ids"].shape[-1]
            text = processor.decode(output[0][input_len:], skip_special_tokens=True)
            print(f"{fn} -> {text}")

        except Exception as e:
            # Don’t crash the whole loop; show which file caused trouble
            print(f"[ERROR] {fn}: {e}")
