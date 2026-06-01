# Stage 1 Verification

## Environment

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL
python -m venv .venv
.\.venv\Scripts\Activate.ps1
cd backend
python -m pip install -e ".[dev]"
cd ..
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Automated Checks

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider backend/tests
```

Expected result: all cube engine and FastAPI endpoint tests pass.

## Backend API

```powershell
cd backend
..\.venv\Scripts\uvicorn.exe app.main:app --reload
```

In another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Expected result: `status` is `ok`.

## Frontend MVP

```powershell
python -m http.server 5173 -d frontend
```

Open `http://localhost:5173`, click `Check API`, then verify:

- `Scramble` returns a valid unsolved cube.
- `Solve` prepares Kociemba two-phase moves through the backend.
- `Play` or `Step Forward` replays the solution to solved.
- Disconnecting the API leaves the UI usable in local fallback mode for tracked move histories.
