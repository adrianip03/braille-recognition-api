# Braille Recognition API

API for detecting Braille characters in images and converting to English text.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('words')"
python main.py

Server runs at http://localhost:8000
```

## API Endpoints
### POST /predict/
Upload image to detect Braille text.

Request: Multipart form with file (image)

Response:

```json
{
    "braille": "таБтаВ таГтаД",
    "text": "Hello world",
    "confidence": 0.95,
    "boundingBox": [0.1, 0.2, 0.8, 0.3],
    "inpaintedImage": {
        "mimeType": "image/png",
        "encoding": "base64",
        "data": "iVBORw0KGgo..."
    }
}
```
### GET /
Health check

### GET /classes/
List detectable character classes

## Usage Examples
### cURL
``` bash
curl -X POST "http://localhost:8000/predict/" \
     -F "file=@image.jpg"
```

## Python
```python
import requests
response = requests.post(
    "http://localhost:8000/predict/",
    files={"file": open("image.jpg", "rb")}
)
print(response.json()["text"])
```

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Invalid file (not an image) |
| 500 | Server/processing error |


## Requirements
Python 3.8+

See requirements.txt for dependencies

