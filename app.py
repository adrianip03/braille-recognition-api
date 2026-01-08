from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import io
import uvicorn
import numpy as np
from scipy import stats
from ultralytics import YOLO

classNameToBraille = {
    'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑',
    'f': '⠋', 'g': '⠛', 'h': '⠓', 'i': '⠊', 'j': '⠚',
    'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝', 'o': '⠕',
    'p': '⠏', 'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞',
    'u': '⠥', 'v': '⠧', 'w': '⠺', 'x': '⠭', 'y': '⠽', 'z': '⠵',
    
    'capital': '⠠',
    
    # this? 
    'number': '⠼',
    
    # Punctuation
    '.': '⠲',    # period
    ',': '⠂',    # comma
    '?': '⠦',    # question mark
    '!': '⠖',    # exclamation mark
    ';': '⠆',    # semicolon
    ':': '⠒',    # colon
    "'": '⠄',    # apostrophe
    '"': '⠐⠂',  # quotation mark
    '-': '⠤',    # hyphen
    '(': '⠶',    # opening parenthesis
    ')': '⠶',    # closing parenthesis (same as opening in Braille)
    '/': '⠌',    # slash
    '[': '⠦',    # opening bracket (same as ? in some systems)
    ']': '⠴',    # closing bracket
    '{': '⠠⠦',  # opening brace
    '}': '⠠⠴',  # closing brace
    '<': '⠪',    # less than
    '>': '⠕',    # greater than (same as o in some systems)
}

numberDict = {
    '0': '⠚', '1': '⠁', '2': '⠃', '3': '⠉', '4': '⠙',
    '5': '⠑', '6': '⠋', '7': '⠛', '8': '⠓', '9': '⠊',
}

app = FastAPI(title="Braille Recognition API")
model = YOLO("best.pt")

@app.get("/")
async def root():
    return {"message": "Braille Recognition API", "status": "running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    contents = await file.read()
    
    results = model(contents)
    
    # process results        
    cord = []
    for i in results[0].summary():
        XY = i['box']
        X = (XY['x1'], XY['x2'])
        Y = (XY['y1'], XY['y2'])
        midpoint = (np.mean(X), np.mean(Y))
        cord.append((i['name'],midpoint))

    if not cord:
        return JSONResponse({
            "braille": "",
            "text": "",
            "message": "No braille characters detected",
            "confidence": 0
        })
            
    xy_sorted_cord = sorted(cord, key=lambda x: (x[1][1], x[1][0]))
    
    horizontal_distance_from_front = []
    for i in range(1, len(cord)): 
        horizontal_distance_from_front.append(np.mean(Y)-xy_sorted_cord[i-1][1][1])
        
    if any(dist < 0 for dist in horizontal_distance_from_front):
        same_line_horizontal_dist = [dist for dist in horizontal_distance_from_front if dist > 0]
    else: 
        same_line_horizontal_dist = horizontal_distance_from_front
        
    # check if same line horizontal dist is bimodal
    # hartigan's diptest
    dip_statistic, p_value = stats.diptest.diptest(np.array(same_line_horizontal_dist))
    if p_value < 0.05: 
        from sklearn.cluster import KMeans
        X = np.array(same_line_horizontal_dist).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X)
        space_threshold = np.mean(kmeans.cluster_centers_)
        space_postions = [i for i, dist in enumerate(horizontal_distance_from_front) if (dist > space_threshold or dist < 0)]
    else: 
        space_postions = [i for i, dist in enumerate(horizontal_distance_from_front) if dist < 0]
    
    processed_braille = ""
    processed_text = ""
    capitalFlag = False
    numberFlag = False
    for idx, cord in enumerate(xy_sorted_cord): 
        if idx in space_postions: 
            processed_braille += " "
            processed_text += " "
        processed_braille += classNameToBraille[cord[0]]
        if cord[0] == "capital": 
            capitalFlag = True
        elif cord[0] == "number": 
            numberFlag = True
        else: 
            if capitalFlag: 
                processed_text += cord[0].upper()
                capitalFlag = False
            elif numberFlag: 
                processed_text += numberDict[cord[0]]
            else: 
                processed_text += cord[0]
    

    return JSONResponse({
        "braille": processed_braille,
        "text": processed_text,
    })

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
    return {"classes": list(model.class_names.values())}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)