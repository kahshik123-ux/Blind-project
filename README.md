# VisionVoice

VisionVoice is a camera-based reading assistant. The existing React/Vite interface and browser speech controls are retained. OCR now uses local PaddleOCR and EasyOCR through FastAPI, compares their text and bounding boxes, and keeps each engine's raw result visible. Browser Tesseract remains a local fallback if the backend is unavailable.

## Requirements

- Windows 10/11 x64
- Python 3.12 x64
- Node.js 18+
- Microsoft Visual C++ 2015–2022 (x64) Redistributable (required by the Paddle native CPU runtime)
- About 2 GB of free disk space for Python packages and OCR model caches
- CPU inference is selected by default. This project does not require CUDA or an NVIDIA GPU.

The verified development machine has Python 3.12.10, 15.7 GB RAM, a 10-core Intel i5-1245U, and Intel integrated graphics. PP-OCRv6 small models are selected to balance CPU throughput and accuracy.

## Install

From the project directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r backend\requirements-cpu.txt
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

The CPU PyTorch wheels come from the official PyTorch CPU wheel index. PaddleOCR, PaddlePaddle, EasyOCR, and the Gemini SDK are installed from PyPI.
If the Microsoft C++ runtime is not installed, install Microsoft's official x64 package:

```powershell
winget install --id Microsoft.VCRedist.2015+.x64 --exact
```

## Download the pretrained OCR models

```powershell
.\.venv\Scripts\python.exe -m backend.download_models
```

The setup command loads each model once so its official pretrained weights are downloaded and cached under `backend\models\`. PaddleOCR is pinned to Paddle's official BOS model host rather than third-party mirrors. The cache is ignored by Git and reused on later launches. The selected PaddleOCR models are `PP-OCRv6_small_det` and `PP-OCRv6_small_rec`, with official page/text orientation classifiers. Page unwarping is disabled for recognition because it can distort close-up words. Camera captures and backend inputs are limited to a 1600-pixel long edge to keep CPU inference responsive; PaddleOCR's orientation classifiers handle rotated text.

## Run

Use two PowerShell terminals:

```powershell
# Terminal 1: local OCR API
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

```powershell
# Terminal 2: React application
npm install
npm run dev -- --host 0.0.0.0 --port 5174
```

Open the Vite URL (usually `http://localhost:5174`). In the development server, `/api` is proxied to FastAPI on port 8000. Enable **Small-text mode (2x center zoom)** for fine print; aim it at the center and capture page sections one at a time.
To view the interface on a phone, connect it to the same network and open `http://<computer-LAN-IP>:5174`. Mobile browsers require HTTPS for camera access on a LAN address; plain HTTP can show the site but will block camera capture. Use a trusted HTTPS development address for camera testing.

## Gemini image verification (optional)

Gemini is disabled until configured. Copy `.env.example` to `.env`, add the key on the backend only, and restart FastAPI:

```powershell
Copy-Item .env.example .env
notepad .env
```

Set `GEMINI_API_KEY` in `.env`. Never put the key in React/Vite variables. The UI requires explicit consent before sending a captured image and OCR text to Google's Gemini API. Gemini suggestions are advisory only; they never overwrite OCR text automatically. If Gemini is unavailable, the local PaddleOCR/EasyOCR result remains usable.

## API

- `GET /api/health` — backend/model readiness without loading models
- `POST /api/ocr/paddle` — PaddleOCR result and confidence
- `POST /api/ocr/easy` — EasyOCR result and confidence
- `POST /api/ocr/combined` — two local OCR results, box-based comparison, and uncertain regions
- `POST /api/ai/analyze` — opt-in Gemini review of the image and OCR output

OCR endpoints accept multipart form data with an `image` file. Images are decoded and processed in memory; the server does not save captured documents. `/api/health` reports the model weights as ready only after the actual model files are present.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s backend\tests -v
.\.venv\Scripts\python.exe -m backend.verify_ocr
npm run build
npm run lint
```

The OCR smoke test uses a generated, local `HELLO` image and runs it through the real `/api/ocr/combined` pipeline and both pretrained models. It does not upload or save an image.

## Model and service licenses

- PaddleOCR and PaddlePaddle are distributed under Apache License 2.0. The official PP-OCRv6 model files are downloaded by PaddleOCR/PaddleX from Paddle's model distribution service; check the model's accompanying terms before redistribution or commercial use.
- EasyOCR is Apache License 2.0. Its pretrained detector/recognizer weights are downloaded from the EasyOCR project's upstream model host; verify the terms shipped with each specific weight before redistributing it.
- PyTorch CPU wheels are distributed under the PyTorch license.
- Gemini is a hosted Google service, subject to Google's Gemini API terms, model availability, quotas, and billing. Gemini image review sends the captured image to Google only after the user opts in.

Model weights are not committed to this repository. No hardware-development components are included.
