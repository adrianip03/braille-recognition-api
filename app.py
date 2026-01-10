from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
import sys
from ultralytics import YOLO
from PIL import Image, ExifTags
import io as image_io

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

try:
    model = YOLO("currentModel.pt")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.get("/")
async def root():
    return {"message": "Braille Recognition API", "status": "running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if model is None: 
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try: 
        contents = await file.read()

        
        image = Image.open(image_io.BytesIO(contents))
        image = fix_image_orientation(image)
        image.show()

        
        results = model(image)
        
        results[0].show()
        
        # process results        
        cords = []
        confidences = []
        
                
        if results[0].boxes is not None and len(results[0].boxes) > 0: 
            boxes = results[0].boxes
                    
            box_coords = []            
                
            for box in boxes: 
                cls_idx = int(box.cls[0])
                class_name = model.names[cls_idx]
                conf = float(box.conf[0])
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                box_coords.append([x1, y1, x2, y2])
                
                midpoint = ((x1+x2)/2, (y1+y2)/2)
                cords.append((class_name, midpoint))
                confidences.append(conf)
            
            box_array = np.array(box_coords)
            
            x1_min = np.min(box_array[:, 0])  
            y1_min = np.min(box_array[:, 1])  
            
            x2_max = np.max(box_array[:, 2])
            y2_max = np.max(box_array[:, 3])
            
            img_width, img_height = image.size
            normalized_bounding_box = [float(x1_min) / img_width, float(y1_min) / img_height, float(x2_max - x1_min) / img_width, float(y2_max - y1_min) / img_height]
                

        if not cords:
            return JSONResponse({
                "braille": "",
                "text": "",
                "confidence": 0, 
                "message": "No braille characters detected",
            })
                
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
            
        
        # for line in sorted_lines:
        #     for char in line:
        #         print(f"{char[0]}", end=" ")
        #     print()
            
        # print()
        
        horizontal_distance_from_front = []
        for line in sorted_lines: 
            x_dist_within_line = [-1]
            for i in range(1, len(line)): 
                x_dist_within_line.append(line[i][1][0] - line[i-1][1][0])
            horizontal_distance_from_front.append(x_dist_within_line)
        
        # for line in horizontal_distance_from_front:
        #     for char in line:
        #         print(f"{char:.2f}", end=" ")
        #     print()
            
        same_line_horizontal_dist = [dist for sublist in horizontal_distance_from_front for dist in sublist if dist > 0]
        

        # try clustering to two clusters
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
            
        # print(space_threshold)
        
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
                        processed_text += numberDict[char[0]]
                        numberFlag = False
                    else: 
                        processed_text += char[0]
        
        avg_confidence = np.mean(confidences) if confidences else 0
        
        return JSONResponse({
            "braille": processed_braille,
            "text": processed_text,
            "confidence": avg_confidence, 
            "boundingBox": normalized_bounding_box
        })
        
    except Exception as e: 
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# @app.post("/predict/visualize/")
# async def predict_visualize(file: UploadFile = File(...)):
#     if not file.content_type.startswith('image/'):
#         raise HTTPException(status_code=400, detail="File must be an image")
    
#     contents = await file.read()
#     annotated_image = model.predict_and_draw(contents)
    
#     return StreamingResponse(
#         io.BytesIO(annotated_image),
#         media_type="image/jpeg",
#         headers={"Content-Disposition": f"attachment; filename=result_{file.filename}"}
#     )

@app.get("/classes/")
async def get_classes():
    return {"classes": model.names}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)