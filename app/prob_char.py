
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from torchvision.ops import nms
from helper import TorchBox

def torch_inference(model, image_list, conf_threshold=0.045, iou_threshold=0.25, input_size = 1024): 
    # 1. Get class names
    class_names = model.names

    # 2. Preprocess & Handle Scaling
    batch_tensors = list()
    original_sizes = list()
    # results = model(image_list, imgsz=input_size, iou=0.25, conf=0.25)
    # true_lengths = [len(result.boxes.xywh.tolist()) for result in results]
    
    for image in image_list: 
        orig_h, orig_w = image.shape[:2]
        original_sizes.append((orig_h, orig_w))

        # YOLO usually resizes the long side to 640 and pads the short side
        r = input_size / max(orig_h, orig_w) # scaling ratio

        # Resize image maintaining aspect ratio
        img_resized = cv2.resize(image, (int(orig_w * r), int(orig_h * r)))

        # Create a blank 640x640 canvas and paste the resized image (Letterboxing)
        canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        canvas[:int(orig_h * r), :int(orig_w * r)] = img_resized

        img_tensor = torch.from_numpy(canvas.transpose((2, 0, 1))).float() / 255.0
        batch_tensors.append(img_tensor)
    
    batch_tensor = torch.stack(batch_tensors).to(model.device)

    # 3. Inference
    with torch.no_grad():
        batch_output = model.model(batch_tensor)[0]

    # 4. Probabilities: Use SOFTMAX instead of Sigmoid
    # This ensures the total probability for each box equals 1.0 (100%)
    # and eliminates the 0.5 neutral bias.
    results = list()
    for idx, output in enumerate(batch_output):
        output = output.transpose(0,1)
        orig_h, orig_w = original_sizes[idx]
        # true_length = true_lengths[idx]
        
        boxes = output[:, :4]
        scores_all = torch.softmax(output[:, 4:], dim=1) 
        max_scores, _ = torch.max(scores_all, dim=1)

        # 5. Filter & NMS
        mask = max_scores > conf_threshold
        f_boxes = boxes[mask]
        f_scores = scores_all[mask]
        f_max_scores = max_scores[mask]

        # Convert [center_x, center_y, w, h] to [x1, y1, x2, y2]
        nms_boxes = f_boxes.clone()
        nms_boxes[:, 0] = f_boxes[:, 0] - f_boxes[:, 2] / 2
        nms_boxes[:, 1] = f_boxes[:, 1] - f_boxes[:, 3] / 2
        nms_boxes[:, 2] = f_boxes[:, 0] + f_boxes[:, 2] / 2
        nms_boxes[:, 3] = f_boxes[:, 1] + f_boxes[:, 3] / 2

        keep_indices = nms(nms_boxes, f_max_scores, iou_threshold=iou_threshold)

        # 6. Final Results with Coordinate Correction
        boxes = list()
        for idx in keep_indices:
            # if (len(boxes) == true_length): 
            #     break
            # Scale coordinates back to original image size
            # We divide by the ratio 'r' used during resizing
            x1, y1, x2, y2 = nms_boxes[idx].tolist()
            
            real_x1 = x1 / r
            real_y1 = y1 / r
            real_x2 = x2 / r
            real_y2 = y2 / r

            prob_dist = f_scores[idx]
            dist_dict = {class_names[i]: float(prob_dist[i]) for i in range(len(class_names))}
            sorted_dist = dict(sorted(dist_dict.items(), key=lambda x: x[1], reverse=True))
            
            real_xyxy = [real_x1, real_y1, real_x2, real_y2]
            
            box = TorchBox(sorted_dist, xyxy=real_xyxy)
            boxes.append(box)
        results.append(boxes)
    
    for result in results:
        print(" ".join([box.class_name for box in result]))
    
    return results