import re
import cv2
import numpy as np

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
