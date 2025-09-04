import torch


MODEL_ID = "google/paligemma2-3b-pt-224" # change to "google/paligemma-3b-pt-224" for paligemma  Visit https://huggingface.co/google/paligemma-3b-pt-224 for more details
DATASET_ID = "hf_username/partial_occlusion-medium_contrast-paligemma"  # change to your dataset from the hub
BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 5e-5
MODEL_DTYPE = torch.bfloat16
MODEL_REVISION = "main"  # change to "bfloat16" for google/paligemma-3b-pt-224
PROMPT = "detect circle; triangle"