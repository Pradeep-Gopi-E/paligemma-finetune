import os
from src.inference.detect import detect_with_paligemma
from configs.inference_config import IMAGE_DIR,  MODEL_ID
from utils.utils import parse_bbox_and_labels, draw_boxes

def test_paligemma_on_folder():
    model_id = MODEL_ID
    image_folder = IMAGE_DIR
    prompt = "detect circle ; triangle"

    results = detect_with_paligemma(model_id, image_folder, prompt)

    output_dir = os.path.join(image_folder, "detections")
    os.makedirs(output_dir, exist_ok=True)

    for res in results:
        if "prediction" not in res:
            continue
        img_path = os.path.join(image_folder, res['file'])
        boxes, labels = parse_bbox_and_labels(res['prediction'])
        
        if len(boxes) > 0:
            save_path = os.path.join(output_dir, f"det_{res['file']}")
            draw_boxes(img_path, boxes, labels, save_path)
            print(f"Saved: {save_path}")
        else:
            print(f"No detections for {res['file']}")

if __name__ == "__main__":
    test_paligemma_on_folder()
