# Stage 2 Dataset Generator

## Scope

The supervised dataset generator creates reproducible cube states for the
first learning baseline. Each record starts from a generated scramble and stores
the inverse move sequence as the target solution.

## Record Format

Each JSONL line contains:

- `record_id`: stable ID for depth and sample index.
- `stickers`: 54-character cube sticker string.
- `sticker_indices`: integer sticker encoding aligned with `FACE_ORDER`.
- `scramble`: moves applied to the solved cube.
- `depth`: scramble length.
- `target_moves`: inverse solution moves.
- `target_action`: first inverse move as the fixed 18-action ID.
- `is_solved`: whether the source state is already solved.
- `seed`: per-record scramble seed.
- `sample_index`: index within the depth bucket.

Generated datasets should be written under `datasets/`, which is ignored by
source control.

## Commands

Generate a small smoke dataset:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
..\.venv\Scripts\python.exe -m rubic_rl.datasets.generator --out ..\datasets\smoke.jsonl --depths 1 2 3 --samples-per-depth 10 --seed 20260531 --include-solved
```

Generate a larger baseline dataset:

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
..\.venv\Scripts\python.exe -m rubic_rl.datasets.generator --out ..\datasets\supervised-depth-1-7.jsonl --depths 1 2 3 4 5 6 7 --samples-per-depth 10000 --seed 20260531
```

## Verification

```powershell
cd D:\WorkSpaces\MyCode\GithubProject\Rubic-RFL\rl
..\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests
```

The tests prove deterministic generation, target move correctness, depth
counts, config validation, and JSONL roundtrip behavior.
