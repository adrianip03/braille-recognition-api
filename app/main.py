from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
import sys
from ultralytics import YOLO
from PIL import Image, ExifTags
import io as image_io
import time
from helper import split_image, get_merged_boxes, non_max_suppression, get_normalized_bounding_box
import traceback 
from pathlib import Path
import os

# Helper function to fix image orientation based on EXIF data
def fix_image_orientation(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation to image if present."""
    try:
        exif = image._getexif()
        if exif is not None:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            
            if orientation in exif:
                orientation_value = exif[orientation]
                
                if orientation_value == 3:
                    image = image.rotate(180, expand=True)
                elif orientation_value == 6:
                    image = image.rotate(270, expand=True)
                elif orientation_value == 8:
                    image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass
    
    return image

def preprocess_image(image: Image.Image) -> Image.Image:    
    
    image = fix_image_orientation(image)
    
    return image

# db clustering to detect lines
def detect_lines_dbscan(y_coords, eps_auto=True, safety_factor=5):
    y_array = np.array(y_coords).reshape(-1, 1)
    
    if eps_auto:
        from sklearn.neighbors import NearestNeighbors
        
        neighbors = NearestNeighbors(n_neighbors=2)
        neighbors_fit = neighbors.fit(y_array)
        distances, indices = neighbors_fit.kneighbors(y_array)
        
        distances = np.sort(distances[:, 1])
        
        # we want epsilon to be max nearest within cluster neighbor distance
        # set to be 95% * safety factor to prevent extreme cases with single point cluster
        eps = np.percentile(distances, 95) * safety_factor
    else:
        eps = 10  # approx char height
    
    dbscan = DBSCAN(eps=eps, min_samples=1)
    labels = dbscan.fit_predict(y_array)
    
    return labels

classNameToBraille = {
    'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑',
    'f': '⠋', 'g': '⠛', 'h': '⠓', 'i': '⠊', 'j': '⠚',
    'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝', 'o': '⠕',
    'p': '⠏', 'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞',
    'u': '⠥', 'v': '⠧', 'w': '⠺', 'x': '⠭', 'y': '⠽', 'z': '⠵',
    'capital': '⠠',
    'number': '⠼',
    'dot_4': '',
    
    # # Punctuation
    # '.': '⠲',    # period
    # ',': '⠂',    # comma
    # '?': '⠦',    # question mark
    # '!': '⠖',    # exclamation mark
    # ';': '⠆',    # semicolon
    # ':': '⠒',    # colon
    # "'": '⠄',    # apostrophe
    # '"': '⠐⠂',  # quotation mark
    # '-': '⠤',    # hyphen
    # '(': '⠶',    # opening parenthesis
    # ')': '⠶',    # closing parenthesis (same as opening in Braille)
    # '/': '⠌',    # slash
    # '[': '⠦',    # opening bracket (same as ? in some systems)
    # ']': '⠴',    # closing bracket
    # '{': '⠠⠦',  # opening brace
    # '}': '⠠⠴',  # closing brace
    # '<': '⠪',    # less than
    # '>': '⠕',    # greater than (same as o in some systems)
}

numberDict = {
    'j': '0', 'a': '1', 'b': '2', 'c': '3', 'd': '4',
    'e': '5', 'f': '6', 'g': '7', 'h': '8', 'i': '9',
}

app = FastAPI(title="Braille Recognition API")

@app.get("/")
async def root():
    return {"message": "Braille Recognition API", "status": "running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Start overall timer
    start_time = time.time()
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Model load
    model_load_start = time.time()
    model = None
    parent_dir = Path(__file__).parent.parent
    model_path = None
    if((parent_dir / "model").exists()):
        model_path = parent_dir / "model" / "currentModel.pt"
    else:
        model_path = Path('/app/model/currentModel.pt')
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e} in directory {os.getcwd()}")
        model = None
    
    if model is None: 
        raise HTTPException(status_code=500, detail="Model not loaded")
    model_load_time = time.time() - model_load_start
    
    try: 
        # File read
        file_read_start = time.time()
        contents = await file.read()
        file_read_time = time.time() - file_read_start
        
        # Image preprocess
        image_process_start = time.time()
        image = Image.open(image_io.BytesIO(contents))
        image = preprocess_image(image)
        image_list = split_image(image)
        image_process_time = time.time() - image_process_start
        
        # Model infernce
        inference_start = time.time()
        results = model(image_list, imgsz=640, iou=0.25, conf=0.25, agnostic_nms = True)
        # results = model(image, imgsz=1024, iou=0.25, conf=0.25)
        inference_time = time.time() - inference_start
        
        # for result in results: 
        #     result.show()
        
        # Post processing 
        # merge box from split images and do nms for overlapping areas
        postprocess_start = time.time()
        merged_boxes = get_merged_boxes(results, image.size)
        boxes = non_max_suppression(merged_boxes, 0.25)
        postprocess_time = time.time() - postprocess_start
            
        if not boxes:
            return JSONResponse({
                "braille": "",
                "text": "",
                "confidence": 0, 
                "message": "No braille characters detected",
                "processing_time_ms": round((time.time() - start_time) * 1000, 2)
            })
            
        # get outer bounding box
        normalized_bounding_box = get_normalized_bounding_box(boxes, image.size)
            
        # get average confidence
        confidences = [float(box.conf[0]) for box in boxes]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        cords = []
        for box in boxes: 
            cls_idx = int(box.cls[0])
            class_name = model.names[cls_idx]
            
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            midpoint = ((x1+x2)/2, (y1+y2)/2)
            cords.append((class_name, midpoint))
        
        # Start line detection timer
        line_detect_start = time.time()
        # group together similar y as same line
        y_cords = [x[1][1] for x in cords]
        line_labels = detect_lines_dbscan(y_cords)
    
        line_groups = {}
        for i, label in enumerate(line_labels):
            if label not in line_groups:
                line_groups[label] = []
            line_groups[label].append(cords[i])
            
        sorted_lines = []
        for label in sorted(line_groups.keys(),  key=lambda l: np.mean([c[1][1] for c in line_groups[l]])):
            line_chars = sorted(line_groups[label], key=lambda x: x[1][0])
            sorted_lines.append(line_chars)
            
        line_detect_time = time.time() - line_detect_start
        
        # Start spacing analysis timer
        spacing_start = time.time()
        horizontal_distance_from_front = []
        for line in sorted_lines: 
            x_dist_within_line = [-1]
            for i in range(1, len(line)): 
                x_dist_within_line.append(line[i][1][0] - line[i-1][1][0])
            horizontal_distance_from_front.append(x_dist_within_line)
        
        same_line_horizontal_dist = [dist for sublist in horizontal_distance_from_front for dist in sublist if dist > 0]
        
        # try clustering into two clusters
        if len(same_line_horizontal_dist) >= 2: 
            X = np.array(same_line_horizontal_dist).reshape(-1, 1)
            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X)
            labels = kmeans.fit_predict(X)
            centers = sorted(kmeans.cluster_centers_.flatten())
            
            # silhouette check: 
            score = silhouette_score(X, labels)
            
            if score > 0.5: 
                space_threshold = np.mean(centers)
            else: 
                space_threshold = sys.maxsize            
        else: 
            space_threshold = sys.maxsize
        spacing_time = time.time() - spacing_start
        
        # Start braille conversion timer
        conversion_start = time.time()
        processed_braille = ""
        processed_text = ""
        capitalFlag = False
        numberFlag = False
        
        for i, line in enumerate(sorted_lines): 
            for j, char in enumerate(line): 
                if (horizontal_distance_from_front[i][j] == -1 and i != 0):
                    # newline? 
                    processed_braille += " "
                    processed_text += " "
                elif (horizontal_distance_from_front[i][j] > space_threshold):
                    processed_braille += " "
                    processed_text += " "
                    
                processed_braille += classNameToBraille[char[0]]
                if char[0] == "capital": 
                    capitalFlag = True
                elif char[0] == "number": 
                    numberFlag = True
                else: 
                    if capitalFlag: 
                        processed_text += char[0].upper()
                        capitalFlag = False
                    elif numberFlag: 
                        if char[0] in numberDict: 
                            processed_text += numberDict[char[0]]
                        numberFlag = False
                    else: 
                        processed_text += char[0]
        conversion_time = time.time() - conversion_start
        
        total_time = time.time() - start_time
        
        timing_details = {
            "total_ms": round(total_time * 1000, 2),
            "model_load_ms": round(model_load_time * 1000, 2),
            "file_read_ms": round(file_read_time * 1000, 2),
            "image_processing_ms": round(image_process_time * 1000, 2),
            "model_inference_ms": round(inference_time * 1000, 2),
            "post_processing_ms": round(postprocess_time * 1000, 2),
            "line_detection_ms": round(line_detect_time * 1000, 2),
            "spacing_analysis_ms": round(spacing_time * 1000, 2),
            "braille_conversion_ms": round(conversion_time * 1000, 2),
            "other_processing_ms": round((total_time - (model_load_time + file_read_time + image_process_time + inference_time + postprocess_time + line_detect_time + spacing_time + conversion_time)) * 1000, 2)
        }
        
        print(timing_details)
        
        return JSONResponse({
            "braille": processed_braille,
            "text": processed_text,
            "confidence": avg_confidence, 
            "boundingBox": normalized_bounding_box
        })
        
    except Exception as e: 
        error_trace = traceback.format_exc()
        print(e)
        print(str(error_trace))
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/classes/")
async def get_classes():
    model = None
    current_dir = Path(__file__).parent
    model_path = None
    if((current_dir / "model").exists()):
        model_path = current_dir / "model" / "currentModel.pt"
    else:
        model_path = Path('/app/model/currentModel.pt')
    try:
        model = YOLO(model_path)
        return {"classes": model.names}
    except Exception as e:
        print(f"Error loading model: {e} in directory {os.getcwd()}")
        model = None

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)