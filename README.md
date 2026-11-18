# AnswerCheck AI — Complete Working Project

This version implements the original project requirements as a local Flask web application.

## Implemented
- Digitize scanned answer sheets using OpenCV + Tesseract OCR
- Student answer extraction from uploaded images
- Teacher master-answer / marking-scheme input
- Keyword coverage analysis
- TF-IDF + cosine similarity using Scikit-learn
- Transparent weighted score calculation
- Automatic marks and grade generation
- Automated feedback
- SQLite storage of every evaluation
- Reusable marking schemes
- Evaluation history with delete
- Dashboard statistics
- REST-style API endpoints
- Responsive education-themed UI using the supplied background images

## Important OCR prerequisite
Tesseract is an external executable in addition to the Python `pytesseract` package.

### Windows
1. Install Tesseract OCR from the official UB Mannheim Windows installer.
2. If Python cannot find it, set the executable path in `app.py`, for example:
   `pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"`

### Install Python packages
```bash
pip install -r requirements.txt
```

### Run
```bash
python app.py
```
Open http://127.0.0.1:5000

## API
- GET `/api/health`
- POST `/api/ocr`
- POST `/api/evaluate`
- GET `/api/evaluations`
- GET `/api/evaluations/<id>`
- DELETE `/api/evaluations/<id>`
- GET `/api/dashboard`
- GET `/api/answer-keys`
- POST `/api/answer-keys`

## Scoring
The demo scoring is intentionally explainable:
- 45% meaningful keyword coverage
- 55% TF-IDF cosine similarity

This is not a replacement for a teacher's judgment. Handwritten answers, valid paraphrases and domain-specific terminology can be missed by simple NLP, so the UI explicitly supports manual review.

## Production extensions
For a production deployment, add authentication/roles, CSRF protection, encrypted storage, object storage for scans, a proper job queue for OCR, stronger semantic embeddings, rubric-level scoring, audit logs and HTTPS.

## Integrated OCR workflow (updated)

The old **Auto Structure** preview page has been replaced by **OCR Workspace**:

- **Question Paper OCR** uploads a scan, runs Tesseract and the question parser, and renders every detected question in editable question cards. Each card includes its detected marks and a `Load in evaluator` action. Subquestions are shown under their parent question.
- **Student Answer Sheet OCR** uploads a scan, parses student name/roll number and numbered answers, automatically fills the Student Name and Roll Number fields in the evaluator, and renders each detected answer in an editable answer card. `Load answer in evaluator` transfers a selected answer into the evaluator.
- OCR output is editable before it is used. This is intentional because OCR/parser output can be imperfect, especially for handwriting, skewed scans, and unusual question numbering.
- The existing scoring engine, marking-scheme storage, evaluation history, and dashboard remain in place.

OCR currently supports image files (JPG/JPEG, PNG, WEBP, BMP and TIFF). PDF extraction is not enabled in this build. OCR is local (Tesseract); no external AI API key is required.
\n\n## Master Answer Key OCR\nUpload a typed/scanned master key from the OCR Workspace. Detected question, model answer and marks are editable; save each entry to reusable Marking Schemes.\n

## Gemini Vision evaluation input\nThe primary grading flow requires **both** the question paper and the student answer sheet (each image or PDF). Gemini reads the question wording and printed marks from the question paper, then matches and grades handwritten answers from the student sheet. Saved marking schemes are optional supplementary guidance. Every AI grade requires teacher review.\n