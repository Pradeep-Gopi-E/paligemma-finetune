import torch


MODEL_ID = "google/paligemma-3b-pt-224"   # base model
DATASET_ID = "godeep/partial_occlusion-medium_contrast-paligemma"  # change to your dataset
BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 5e-5
MODEL_DTYPE = torch.bfloat16
MODEL_REVISION = "bfloat16"
PROMPT = "detect circle; triangle"