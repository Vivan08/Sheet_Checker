# AnswerCheck AI — Mac + Windows

This archive excludes the bundled Windows `.venv` because virtual environments are OS/Python-version specific. It retains project code, static/templates, uploads, and your supplied `answercheck.db`.

## macOS
1. Install Homebrew (https://brew.sh).
2. In Terminal, inside this folder run `chmod +x setup_mac.sh run_mac.sh`, then `./setup_mac.sh`.
   It installs Tesseract and Python dependencies into `.venv` and starts Flask.
3. Later run `./run_mac.sh`, or in VS Code terminal:
   `source .venv/bin/activate` then `python app.py`.
4. Visit http://127.0.0.1:5000

The app discovers Tesseract through PATH on macOS/Linux and uses the standard install path on Windows. On POSIX startup it restores owner write/traverse permissions on project folder, uploads, and existing DB without deleting data. Keep the folder owned by your account.

## Windows
Keep your existing working environment if preferred. For a clean setup install Tesseract (https://github.com/UB-Mannheim/tesseract/wiki), run `setup_windows.bat` once, then `run_windows.bat`. Windows Tesseract path detection is retained.

## Data
`answercheck.db` is from your uploaded DB; existing uploads are retained. Back up the DB before major changes.


## Gemini Vision complete-paper grading (new)
This feature sends the **original student paper image/PDF directly to Gemini**. It does not use Tesseract or TF-IDF similarity for this grading path. Results are provisional and must be reviewed by a teacher.

### 1) Create a Gemini API key
1. Open Google AI Studio: https://aistudio.google.com/app/apikey
2. Sign in and create an API key. Keep it private; do not paste it into the website or commit it to Git.

### 2) Add the key to this project
1. In the project root (same folder as `app.py`), copy `.env.example` to a new file named `.env`.
2. Open `.env` and replace `PASTE_YOUR_API_KEY_HERE` with your key:
   `GEMINI_API_KEY=your_actual_key_here`
3. Optional: change `GEMINI_MODEL=gemini-2.5-flash` if you want to use another model available to your API account.
4. `.env` is ignored by Git. Never share it or upload it publicly.

### 3) Install dependencies and restart
macOS: activate `.venv`, then run `pip install -r requirements.txt`.
Windows: activate `.venv` (or rerun `setup_windows.bat`) and run `pip install -r requirements.txt`.
Restart Flask after editing `.env`.

### 4) Use it
1. Open AnswerCheck AI and save marking schemes in **Answer Keys** first.
2. In **Gemini Vision — Grade a Complete Paper**, upload a clear JPG/PNG or PDF of the full answer sheet.
3. Select one saved marking scheme, or leave the dropdown on **Use all saved answer keys**.
4. Click **Grade complete paper with Gemini** and inspect every question, transcription, score, and feedback.
5. Verify name/roll, question mapping, and marks manually before accepting results.

**Important:** Vision grading can still misread handwriting or make grading mistakes. The teacher review flag is intentional; Gemini output is not an automatically finalized grade.


## Gemini Vision evaluation input\nThe primary grading flow requires **both** the question paper and the student answer sheet (each image or PDF). Gemini reads the question wording and printed marks from the question paper, then matches and grades handwritten answers from the student sheet. Saved marking schemes are optional supplementary guidance. Every AI grade requires teacher review.\n