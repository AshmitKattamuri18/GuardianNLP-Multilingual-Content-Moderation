# GuardianNLP - Multilingual Content Moderation

GuardianNLP is a multilingual content moderation application that uses deep learning models to detect toxic and abusive content.

The project includes a React/Vite frontend, a FastAPI inference backend, and two trained models: IndicGuard-Lite and IndicGuard-MuRIL.

An optional negation-aware candidate is included and disabled by default.

## 1. Project Structure

```text
GuardianNLP-Multilingual-Content-Moderation/
|
|-- backend/
|   |-- main.py
|   |-- indicguard_runtime.py
|   |-- requirements.txt
|   |-- .env.example
|
|-- frontend/
|   |-- src/
|   |-- package.json
|   |-- package-lock.json
|
|-- guardian_nlp_dual_export/
|   |-- model_finetuned/
|   |-- model_scratch/
|   |-- ensemble_config.json
|   |-- class_labels.json
|   |-- preprocessing.json
|
|-- README.md
|-- .gitignore
```

## 2. Prerequisites

Install the following software:

- Python 3.11
- Node.js and npm
- Git

## 3. Clone the Repository

Open PowerShell and run:

```powershell
git clone https://github.com/AshmitKattamuri18/GuardianNLP-Multilingual-Content-Moderation.git

cd GuardianNLP-Multilingual-Content-Moderation
```

## 4. Download Trained Models

The trained model weights are stored separately on Google Drive because of GitHub file size limitations.

Download the model archive:

**[GuardianNLP Trained Models - Google Drive](https://drive.google.com/file/d/1LozQ_ENyV2pFCZZ3K8rffFCr8LnKmIJQ/view?usp=drive_link)**

Extract the ZIP file.

Copy the extracted `guardian_nlp_dual_export` folder into the repository root, merging it with the existing folder.

After extraction, verify that these files exist:

```text
guardian_nlp_dual_export/
|
|-- model_finetuned/
|   |-- model_state.pt
|   |-- config.json
|   |-- tokenizer.json
|
|-- model_scratch/
|   |-- model.safetensors
|   |-- config.json
|   |-- tokenizer.json
```

Do not place the extracted folder inside another `guardian_nlp_dual_export` folder.

## 5. Backend Setup

Open PowerShell in the repository directory.

Navigate to the backend:

```powershell
cd backend
```

Create a Python virtual environment:

```powershell
py -3.11 -m venv .venv
```

Upgrade pip:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Install backend dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

A compatible PyTorch installation is required. If the installation fails because of PyTorch compatibility, install a PyTorch build appropriate for your operating system and hardware.

### Start the Backend

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

The backend uses the model assets located in `guardian_nlp_dual_export`.

### Optional Negation-Aware Candidate

The optional negation-aware candidate is disabled by default.

To enable it, set the environment variable before starting the backend:

```powershell
$env:ENABLE_NEGATION_CANDIDATE = "true"

.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

## 6. Frontend Setup

Open a second PowerShell terminal.

Navigate to the frontend directory:

```powershell
cd frontend
```

Install dependencies:

```powershell
npm ci
```

Configure the backend URL:

```powershell
$env:VITE_API_BASE_URL = "http://127.0.0.1:8000"
```

Start the frontend:

```powershell
npm run dev -- --host 127.0.0.1 --port 5173
```

Open the application in your browser:

`http://127.0.0.1:5173`

Keep both the backend and frontend terminals running.

## 7. Application Components

**IndicGuard-Lite**

The lightweight model used for multilingual content moderation.

**IndicGuard-MuRIL**

The fine-tuned MuRIL-based model used for multilingual content moderation.

**Inference Backend**

The FastAPI service loads the model assets and processes requests from the frontend.

**Frontend**

The React/Vite interface allows users to interact with the moderation application.

## 8. Development Checks

Backend tests can be run after installing the necessary test dependencies:

```powershell
.\backend\.venv\Scripts\python.exe -m pytest -q backend
```

Frontend checks:

```powershell
cd frontend
npm run lint
npm run build
```

## 9. Important Notes

- Trained model weights are not included in the GitHub repository.
- Download the model archive from Google Drive before running inference.
- Python virtual environments and Node.js dependencies must be installed locally.
- Training datasets and private environment files are not included.
- The application is intended to run locally using the provided backend and frontend.

## 10. Repository

**GitHub:** https://github.com/AshmitKattamuri18/GuardianNLP-Multilingual-Content-Moderation

**Trained Models:** https://drive.google.com/file/d/1LozQ_ENyV2pFCZZ3K8rffFCr8LnKmIJQ/view?usp=drive_link