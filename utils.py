import re
import cv2
import numpy as np
from typing import List, Tuple


def parse_bbox_and_labels(detokenized_output: str):
    box_strings = detokenized_output.split(" ; ")
    labels, boxes = [], []
    fmt = lambda x: float(x) / 1024.0  # Normalize to 1024x1024 size

    for box_str in box_strings:
        matches = re.finditer(
            r'<loc(?P<y0>\d\d\d\d)><loc(?P<x0>\d\d\d\d)><loc(?P<y1>\d\d\d\d)><loc(?P<x1>\d\d\d\d)> (?P<label>.+?)$',
            box_str.strip(),
        )
        for m in matches:
            d = m.groupdict()
            boxes.append([fmt(d['x0']), fmt(d['y0']), fmt(d['x1']), fmt(d['y1'])])  # (x0,y0,x1,y1)
            labels.append(d['label'])
    
    return np.array(boxes), np.array(labels)

def draw_boxes(image_path, boxes, labels, save_path=None):
    img = cv2.imread(image_path)
    H, W = img.shape[:2]

    for (x0, y0, x1, y1), label in zip(boxes, labels):
        x0_abs, y0_abs = int(x0 * W), int(y0 * H)
        x1_abs, y1_abs = int(x1 * W), int(y1 * H)

        cv2.rectangle(img, (x0_abs, y0_abs), (x1_abs, y1_abs), (0, 255, 0), 2)
        cv2.putText(img, label, (x0_abs, max(20, y0_abs - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if save_path:
        cv2.imwrite(save_path, img)
    else:
        cv2.imshow("Detections", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def read_yolo_label_file(file_path):
    """
    Reads YOLO label file (class x_center y_center width height)
    Returns boxes as [x0, y0, x1, y1] normalized and labels
    """
    boxes = []
    labels = []
    class_map = {0: "triangle", 1: "circle"}  # Adjust according to your dataset
    with open(file_path, 'r') as f:
        for line in f.readlines():
            if line.strip() == "":
                continue
            parts = line.strip().split()
            cls_id = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:])
            x0 = x_c - w / 2
            y0 = y_c - h / 2
            x1 = x_c + w / 2
            y1 = y_c + h / 2
            boxes.append([x0, y0, x1, y1])
            labels.append(class_map.get(cls_id, str(cls_id)))
    return np.array(boxes), np.array(labels)

def compute_iou(boxA, boxB):
    """
    Computes IoU between two boxes [x0, y0, x1, y1] in pixels
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    iou = interArea / (boxAArea + boxBArea - interArea + 1e-6)
    return iou

def convert_corners_to_center(box):
    """
    Converts a box from [x_min, y_min, x_max, y_max] to [x_center, y_center, width, height]
    """
    x_min, y_min, x_max, y_max = box
    width = x_max - x_min
    height = y_max - y_min
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    return [x_center, y_center, width, height]

def scale_normalized_boxes(boxes, width, height):
    """
    Scale normalized boxes [x0, y0, x1, y1] to pixel coordinates.
    Returns list of boxes in pixel coordinates.
    """
    return [[b[0] * width, b[1] * height, b[2] * width, b[3] * height] for b in boxes]

def draw_boxes_on_image(image, boxes, labels, color=(0, 255, 0), thickness=2):
    """
    Draws boxes and labels on the image.
    """
    for (x0, y0, x1, y1), label in zip(boxes, labels):
        cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), color, thickness)
        cv2.putText(image, label, (int(x0), max(20, int(y0) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)