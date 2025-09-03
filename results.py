import os
import cv2
import numpy as np
from detect import detect_with_paligemma
from utils import parse_bbox_and_labels, compute_iou, scale_normalized_boxes, draw_boxes_on_image, read_yolo_label_file
from collections import defaultdict

def compute_ap(precisions, recalls):
    """
    Compute Average Precision (AP) using 11-point interpolation as in Pascal VOC.
    """
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        prec_at_recall = [p for p, r in zip(precisions, recalls) if r >= t]
        prec = max(prec_at_recall) if prec_at_recall else 0
        ap += prec
    return ap / 11

def evaluate_map(image_dir, label_dir, model_id, iou_threshold=0.5, visualize=False):
    """
    Evaluate Mean IoU, Precision, Recall, F1-score, and mAP for Paligemma predictions.
    """
    prompt = "detect circle ; triangle"
    raw_predictions = detect_with_paligemma(model_id, image_dir, prompt)
    prediction_outputs = {}

    # Organize predictions
    for entry in raw_predictions:
        if "file" in entry and "prediction" in entry:
            prediction_outputs[entry["file"]] = entry["prediction"]
        elif "file" in entry and "error" in entry:
            print(f"[ERROR] {entry['file']}: {entry['error']}")
            prediction_outputs[entry["file"]] = ""
        else:
            raise TypeError("detect_with_paligemma returned unexpected format")

    iou_scores = []
    all_tp = all_fp = all_fn = 0
    class_tp = defaultdict(list)
    class_fp = defaultdict(list)
    class_fn = defaultdict(int)

    # Loop over each image
    for img_name, prediction_output in prediction_outputs.items():
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            print(f"[WARNING] Image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        # Parse predictions
        pred_boxes_norm, pred_labels = parse_bbox_and_labels(prediction_output)
        pred_boxes_pixels = scale_normalized_boxes(pred_boxes_norm, width, height)

        # Load ground-truth boxes
        base_name = os.path.splitext(img_name)[0]
        label_file = os.path.join(label_dir, base_name + ".txt")
        if not os.path.exists(label_file):
            print(f"[WARNING] No GT for {img_name}")
            continue
        gt_boxes_norm, gt_labels = read_yolo_label_file(label_file)
        gt_boxes_pixels = scale_normalized_boxes(gt_boxes_norm, width, height)

        img_ious = []
        matched_gt = set()
        true_pos = false_pos = 0

        # Match predictions with ground-truth boxes
        for i, pred_box in enumerate(pred_boxes_pixels):
            best_iou = 0
            best_gt_idx = -1
            for j, gt_box in enumerate(gt_boxes_pixels):
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            pred_class = pred_labels[i]

            if best_iou >= iou_threshold and pred_class == gt_labels[best_gt_idx] and best_gt_idx not in matched_gt:
                true_pos += 1
                matched_gt.add(best_gt_idx)
                class_tp[pred_class].append(1)
                class_fp[pred_class].append(0)
            else:
                false_pos += 1
                class_tp[pred_class].append(0)
                class_fp[pred_class].append(1)

            img_ious.append(best_iou)

        # Count false negatives (missed GT boxes)
        for j, gt_class in enumerate(gt_labels):
            if j not in matched_gt:
                class_fn[gt_class] += 1

        false_neg = len(gt_boxes_pixels) - len(matched_gt)
        iou_scores.extend(img_ious)
        all_tp += true_pos
        all_fp += false_pos
        all_fn += false_neg

        # Visualization (optional)
        if visualize:
            draw_boxes_on_image(img, gt_boxes_pixels, gt_labels, color=(0, 255, 0))  # GT = green
            draw_boxes_on_image(img, pred_boxes_pixels, pred_labels, color=(0, 0, 255))  # Pred = red
            cv2.imshow("Comparison", img)
            cv2.waitKey(0)

    # Overall metrics
    mean_iou = np.mean(iou_scores) if iou_scores else 0.0
    precision = all_tp / (all_tp + all_fp + 1e-6)
    recall = all_tp / (all_tp + all_fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    # Compute AP per class and mAP
    ap_per_class = {}
    for cls in class_tp.keys():
        precisions = []
        recalls = []
        tp_cum = np.cumsum(class_tp[cls])
        fp_cum = np.cumsum(class_fp[cls])
        total_gt = tp_cum[-1] + class_fn[cls] if cls in class_fn else tp_cum[-1]
        for tp, fp in zip(tp_cum, fp_cum):
            p = tp / (tp + fp + 1e-6)
            r = tp / (total_gt + 1e-6)
            precisions.append(p)
            recalls.append(r)
        ap_per_class[cls] = compute_ap(precisions, recalls)

    mAP = np.mean(list(ap_per_class.values())) if ap_per_class else 0.0

    # Print results
    print("\n==== Overall Results ====")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, mAP@0.5: {mAP:.4f}")
    print(f"TP: {all_tp}, FP: {all_fp}, FN: {all_fn}")

    return {
        "mean_iou": mean_iou,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mAP@0.5": mAP,
        "true_positives": all_tp,
        "false_positives": all_fp,
        "false_negatives": all_fn,
        "AP_per_class": ap_per_class
    }

# ----------------- Run Test -----------------
if __name__ == "__main__":
    IMAGE_DIR = r"C:\Users\prade\OneDrive\Documents\Manav\Pradeep\Synth-RT-DETR-DATASET\test\partial_occlusion\low_contrast\images"
    LABEL_DIR = r"C:\Users\prade\OneDrive\Documents\Manav\Pradeep\Synth-RT-DETR-DATASET\test\partial_occlusion\low_contrast\labels"
    MODEL_ID = "godeep/paligemma2-finetuned_448"  # change to your fine-tuned model
    IOU_THRESHOLD = 0.5

    results = evaluate_map(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        model_id=MODEL_ID,
        iou_threshold=IOU_THRESHOLD,
        visualize=False  # Set True to see images
    )
