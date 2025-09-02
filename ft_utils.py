import torch

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

