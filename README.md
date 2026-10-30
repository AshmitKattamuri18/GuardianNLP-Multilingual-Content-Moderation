# GuardianNLP — clean local application

This copy contains the React/Vite UI, FastAPI inference service, existing IndicGuard-Lite and MuRIL inference assets, and the optional negation-aware candidate. It is configured to run locally with the candidate disabled by default. No training data or secrets are included.

## Start the backend (PowerShell)

```powershell
cd C:\Users\kashm\GuardianNLP_Final_Clean\backend
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

The local `.env.example` documents settings. The backend finds `guardian_nlp_dual_export` by the default relative path. Set `ENABLE_NEGATION_CANDIDATE=true` in the process environment before starting to opt in; it defaults to `false`.

## Start the frontend (PowerShell)

```powershell
cd C:\Users\kashm\GuardianNLP_Final_Clean\frontend
$env:VITE_API_BASE_URL = 'http://127.0.0.1:8000'
npm run dev -- --host 127.0.0.1
```

Open the Vite URL printed by the command (normally `http://127.0.0.1:5173`). The frontend dependencies are already copied; `package-lock.json` is retained for reinstalling with `npm ci` if needed.

**Machine note:** This copy ran on the current Windows machine with the copied `backend/.venv`; its `pyvenv.cfg` records the CPython 3.11 base interpreter. Node.js/npm and that Python base installation are system prerequisites. The app does not read models from the original project or a global model cache. To move this source copy to another computer, install CPython 3.11 and Node.js, recreate the backend environment from `backend/requirements.txt` (install a suitable PyTorch build), and run `npm ci` in `frontend`.

## Checks

```powershell
cd C:\Users\kashm\GuardianNLP_Final_Clean
.\backend\.venv\Scripts\python.exe -m pytest -q backend
cd frontend
npm run lint
npm run build
```
