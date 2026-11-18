import os
import json
import re
import sqlite3
import shutil
import stat
import sys
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from google import genai
    from google.genai import types as genai_types
except Exception:
    genai = None
    genai_types = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    import pytesseract
    # Windows: standard installer path; macOS/Linux: discover via PATH.
    if os.name == "nt":
        windows_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(windows_tesseract):
            pytesseract.pytesseract.tesseract_cmd = windows_tesseract
    else:
        discovered_tesseract = shutil.which("tesseract")
        if discovered_tesseract:
            pytesseract.pytesseract.tesseract_cmd = discovered_tesseract
except Exception:
    pytesseract = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None

try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = {
        "the","a","an","is","are","was","were","to","of","in","on","for","and",
        "or","with","by","as","at","from","this","that","it","be","can","using",
        "used","which","into","their","its","than","also","has","have","had"
    }

BASE = Path(__file__).resolve().parent
try:
    load_dotenv(BASE / ".env", override=False)
except Exception:
    pass
UPLOAD_DIR = BASE / "uploads"
DB_PATH = BASE / "answercheck.db"
ALLOWED = {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "pdf"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def prepare_storage():
    """Restore owner write/traverse bits on POSIX, without touching stored data."""
    if os.name == "nt":
        return
    try:
        mode = stat.S_IMODE(BASE.stat().st_mode)
        BASE.chmod(mode | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        mode = stat.S_IMODE(UPLOAD_DIR.stat().st_mode)
        UPLOAD_DIR.chmod(mode | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        if DB_PATH.exists():
            mode = stat.S_IMODE(DB_PATH.stat().st_mode)
            DB_PATH.chmod(mode | stat.S_IRUSR | stat.S_IWUSR)
    except OSError as exc:
        print(f"[Storage warning] Could not adjust permissions: {exc}", file=sys.stderr)
        print("Ensure the project folder is owned by your account and writable.", file=sys.stderr)


def db():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            roll_number TEXT,
            question TEXT NOT NULL,
            student_answer TEXT NOT NULL,
            master_answer TEXT NOT NULL,
            max_marks REAL NOT NULL,
            keyword_score REAL NOT NULL,
            similarity_score REAL NOT NULL,
            final_score REAL NOT NULL,
            percentage REAL NOT NULL,
            grade TEXT NOT NULL,
            feedback TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'text',
            image_filename TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS answer_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            question TEXT NOT NULL,
            master_answer TEXT NOT NULL,
            keywords TEXT NOT NULL,
            max_marks REAL NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS gemini_paper_details (
            evaluation_id INTEGER PRIMARY KEY,
            result_json TEXT NOT NULL,
            FOREIGN KEY(evaluation_id) REFERENCES evaluations(id) ON DELETE CASCADE
        );
        """)


def clean_text(text):
    text = text or ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokens(text):
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]*", (text or "").lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def keyword_set(text):
    return set(tokens(text))


def keyword_score(answer, master):
    """Recall of meaningful master-answer terms in the student answer."""
    master_words = keyword_set(master)
    answer_words = keyword_set(answer)
    if not master_words:
        return 0.0
    return len(master_words & answer_words) / len(master_words)


def semantic_score(answer, master):
    """TF-IDF cosine similarity; deterministic and local."""
    if not answer.strip() or not master.strip():
        return 0.0
    if TfidfVectorizer is None:
        return keyword_score(answer, master)
    try:
        matrix = TfidfVectorizer(ngram_range=(1, 2), lowercase=True).fit_transform([answer, master])
        return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    except Exception:
        return keyword_score(answer, master)


def evaluate_answer(answer, master, max_marks):
    k = keyword_score(answer, master)
    s = semantic_score(answer, master)
    # Balanced, transparent heuristic: keyword coverage + TF-IDF similarity.
    combined = (0.45 * k) + (0.55 * s)
    score = round(max_marks * min(1.0, combined), 2)
    pct = round((score / max_marks) * 100, 1) if max_marks else 0
    grade = "A" if pct >= 85 else "B" if pct >= 70 else "C" if pct >= 55 else "D" if pct >= 40 else "Needs Review"

    if pct >= 85:
        feedback = "Strong alignment with the master answer. The response covers most key concepts."
    elif pct >= 70:
        feedback = "Good alignment. A few concepts or supporting details may be missing."
    elif pct >= 55:
        feedback = "Moderate alignment. Review the response for missing concepts and context."
    else:
        feedback = "Low textual alignment. Manual review is recommended before final grading."

    return {
        "keyword_score": round(k * 100, 1),
        "similarity_score": round(s * 100, 1),
        "final_score": score,
        "percentage": pct,
        "grade": grade,
        "feedback": feedback
    }


def preprocess_image(path):
    if cv2 is None:
        return None
    img = cv2.imread(str(path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # Otsu threshold works well for clean scanned pages.
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def extract_ocr(path):
    if pytesseract is None:
        raise RuntimeError("pytesseract is not installed. Run: pip install -r requirements.txt")
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        raise RuntimeError("PDF OCR is not enabled in this local build. Upload a JPG/PNG scan, or add pdf2image + Poppler.")
    processed = preprocess_image(path)
    if processed is None:
        raise RuntimeError("The uploaded image could not be read.")
    config = "--oem 3 --psm 6"
    text = pytesseract.image_to_string(processed, config=config)
    # Keep line breaks: they are essential for detecting question numbers and answer boundaries.
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip()).strip()
    if not text:
        raise RuntimeError("OCR could not detect readable text in the image.")
    return text


# ---------------- DOCUMENT STRUCTURING (v1: OCR + deterministic parsing) ----------------
QUESTION_START = re.compile(
    r"^\s*(?:Q(?:UESTION)?\s*)?(\d{1,2})\s*(?:[.)]|[:\-])\s*(.*)$",
    re.IGNORECASE
)
SUBQUESTION_START = re.compile(r"^\s*\(?([a-z])\)?\s*[.)]\s*(.*)$", re.IGNORECASE)
MARKS_PATTERN = re.compile(r"\[?\(?\s*(\d+(?:\.\d+)?)\s*(?:marks?|m)\s*\)?\]?", re.IGNORECASE)


def parse_question_paper(text):
    """
    Parse both numbered papers (Q1., 1), etc.) and labelled single-question
    documents such as 'Question: ...' / 'Master Answer Key'.
    """
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    questions, current = [], None
    labelled_question_seen = False
    answer_key_heading = any(re.search(r"\bmaster\s+answer\s+key\b|\banswer\s+key\b", x, re.I) for x in lines)

    def finish():
        nonlocal current
        if current:
            current["question"] = re.sub(r"\s+", " ", current["question"]).strip()
            current["answer_key"] = re.sub(r"\s+", " ", current.get("answer_key", "")).strip()
            questions.append(current)
            current = None

    for line in lines:
        # Labelled question: "Question: Explain..." (not "Question 1:")
        labelled = re.match(r"^\s*question\s*:\s*(.+)$", line, re.I)
        numbered = QUESTION_START.match(line)
        if labelled:
            finish()
            labelled_question_seen = True
            content = labelled.group(1).strip()
            mm = MARKS_PATTERN.search(content)
            current = {"question_number": "Q1", "question": MARKS_PATTERN.sub("", content).strip(),
                       "max_marks": float(mm.group(1)) if mm else None, "subquestions": [], "answer_key": ""}
            continue
        if numbered:
            finish()
            number, content = numbered.groups()
            mm = MARKS_PATTERN.search(line)
            current = {"question_number": f"Q{number}", "question": MARKS_PATTERN.sub("", content).strip(),
                       "max_marks": float(mm.group(1)) if mm else None, "subquestions": [], "answer_key": ""}
            continue
        sub = SUBQUESTION_START.match(line)
        if current and sub:
            label, content = sub.groups()
            mm = MARKS_PATTERN.search(line)
            current["subquestions"].append({"label": label.lower(), "question": MARKS_PATTERN.sub("", content).strip(),
                                            "max_marks": float(mm.group(1)) if mm else None})
        elif current:
            # In an answer-key document, content after the question is the model answer.
            if answer_key_heading and current.get("question"):
                current["answer_key"] = (current.get("answer_key", "") + " " + line).strip()
            else:
                current["question"] = (current["question"] + " " + line).strip()
    finish()
    return questions


def parse_answer_sheet(text):
    """Extract student identity and answers from numbered or unnumbered answer sheets."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    joined = "\n".join(lines)
    name, roll = "", ""

    # Also supports title headings: "Student Answer - Priya Nair"
    for line in lines[:25]:
        heading_name = re.match(r"^\s*student\s+answer\s*[-:]\s*(.+)$", line, re.I)
        name_match = re.search(r"(?:student\s*name|name)\s*[:\-]?\s*(.+)$", line, re.I)
        roll_match = re.search(r"(?:roll\s*(?:no|number)?|enrol(?:ment)?\s*no|registration\s*no)\s*[:\-]?\s*([A-Z0-9\-/]+)", line, re.I)
        if heading_name and not name:
            name = heading_name.group(1).strip()
        elif name_match and not name:
            name = name_match.group(1).strip()
        if roll_match and not roll:
            roll = roll_match.group(1).strip()

    start_re = re.compile(r"^\s*(?:Q(?:UESTION)?\s*)?(\d{1,2})\s*(?:[.)]|\s*\(\s*[a-z]\s*\)\s*[.)]?)\s*(.*)$", re.I)
    answers, current = [], None
    body_started = False
    for line in lines:
        # Skip document title and identity labels from answer body
        if re.match(r"^\s*student\s+answer\s*[-:]", line, re.I) or re.match(r"^\s*(?:student\s*)?name\s*:", line, re.I) or re.match(r"^\s*roll\s*(?:no|number)", line, re.I):
            continue
        m = start_re.match(line)
        if m:
            if current:
                answers.append(current)
            current = {"question_number": f"Q{m.group(1)}", "answer": m.group(2).strip()}
            body_started = True
        elif current:
            current["answer"] = (current["answer"] + "\n" + line).strip()
        elif not body_started:
            # Unnumbered answer sheet: treat remaining body as answer to Q1.
            current = {"question_number": "Q1", "answer": line}
            body_started = True
        else:
            current["answer"] = (current["answer"] + "\n" + line).strip()
    if current:
        answers.append(current)
    return {"student_name": name, "roll_number": roll, "answers": answers,
            "unsegmented_text": joined, "review_required": not bool(name and roll) or not bool(answers)}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


@app.errorhandler(sqlite3.OperationalError)
def handle_sqlite_operational_error(error):
    message = str(error)
    if "readonly" in message.lower() or "read-only" in message.lower():
        return jsonify({
            "error": "The database or its folder is read-only. Ensure this project folder is writable, then restart the app.",
            "database": str(DB_PATH)
        }), 500
    app.logger.exception("SQLite operational error")
    return jsonify({"error": "A database operation failed. Check the Flask terminal for details."}), 500


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "ocr_library": pytesseract is not None,
        "opencv": cv2 is not None,
        "scikit_learn": TfidfVectorizer is not None,
        "database": True
    })


@app.post("/api/ocr")
def ocr():
    if "answer_sheet" not in request.files:
        return jsonify({"error": "Please upload an answer-sheet image."}), 400
    file = request.files["answer_sheet"]
    if not file.filename:
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use JPG, PNG, WEBP, BMP or TIFF."}), 400

    safe = secure_filename(file.filename)
    stamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    filename = f"{stamp}_{safe}"
    path = UPLOAD_DIR / filename
    file.save(path)

    try:
        text = extract_ocr(path)
        return jsonify({"text": text, "filename": filename})
    except Exception as exc:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        return jsonify({"error": str(exc)}), 422


@app.post("/api/structure/question-paper")
def structure_question_paper():
    file = request.files.get("document")
    if not file or not file.filename:
        return jsonify({"error": "Upload a question-paper image first."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use JPG, PNG, WEBP, BMP or TIFF."}), 400
    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{secure_filename(file.filename)}"
    path = UPLOAD_DIR / filename
    file.save(path)
    try:
        text = extract_ocr(path)
        questions = parse_question_paper(text)
        return jsonify({"filename": filename, "raw_text": text, "questions": questions,
                        "review_required": not bool(questions) or any(not q["question"] for q in questions)})
    except Exception as exc:
        path.unlink(missing_ok=True)
        return jsonify({"error": str(exc)}), 422


@app.post("/api/structure/answer-sheet")
def structure_answer_sheet():
    file = request.files.get("document")
    if not file or not file.filename:
        return jsonify({"error": "Upload an answer-sheet image first."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use JPG, PNG, WEBP, BMP or TIFF."}), 400
    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{secure_filename(file.filename)}"
    path = UPLOAD_DIR / filename
    file.save(path)
    try:
        text = extract_ocr(path)
        structured = parse_answer_sheet(text)
        structured.update({"filename": filename, "raw_text": text})
        return jsonify(structured)
    except Exception as exc:
        path.unlink(missing_ok=True)
        return jsonify({"error": str(exc)}), 422


@app.post("/api/structure/master-key")
def structure_master_key():
    if "document" not in request.files:
        return jsonify({"error": "No document uploaded."}), 400
    file = request.files["document"]
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Upload an image."}), 400
    filename = secure_filename(file.filename)
    saved_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], saved_name)
    file.save(path)
    try:
        raw_text = extract_ocr(path)
        parsed = parse_question_paper(raw_text)
        # Single labelled answer-key documents may have one question and its answer.
        items = [{"question_number": q.get("question_number", f"Q{i+1}"),
                  "question": q.get("question", ""), "master_answer": q.get("answer_key", ""),
                  "max_marks": q.get("max_marks") or 10, "subquestions": q.get("subquestions", [])}
                 for i, q in enumerate(parsed)]
        return jsonify({"filename": saved_name, "raw_text": raw_text, "answer_keys": items})
    except Exception as exc:
        try: os.remove(path)
        except OSError: pass
        return jsonify({"error": str(exc)}), 422


@app.post("/api/evaluate")
def api_evaluate():
    data = request.form if request.form else request.get_json(silent=True) or {}
    student = clean_text(data.get("student_name", "Anonymous")) or "Anonymous"
    roll = clean_text(data.get("roll_number", ""))
    question = clean_text(data.get("question", ""))
    answer = clean_text(data.get("student_answer", data.get("answer", "")))
    master = clean_text(data.get("master_answer", data.get("answer_key", "")))
    try:
        max_marks = float(data.get("max_marks", data.get("max_score", 10)))
    except Exception:
        max_marks = 10

    if not question or not answer or not master:
        return jsonify({"error": "Question, student answer and master answer are required."}), 400
    if max_marks <= 0:
        return jsonify({"error": "Maximum marks must be greater than zero."}), 400

    result = evaluate_answer(answer, master, max_marks)
    source = clean_text(data.get("source", "text")) or "text"
    image_filename = clean_text(data.get("image_filename", ""))

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with db() as con:
        cur = con.execute("""
            INSERT INTO evaluations (
                student_name, roll_number, question, student_answer, master_answer,
                max_marks, keyword_score, similarity_score, final_score, percentage,
                grade, feedback, source, image_filename, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            student, roll, question, answer, master, max_marks,
            result["keyword_score"], result["similarity_score"], result["final_score"],
            result["percentage"], result["grade"], result["feedback"],
            source, image_filename, now
        ))
        evaluation_id = cur.lastrowid

    return jsonify({"id": evaluation_id, **result, "student": student, "created_at": now})



@app.post("/api/evaluate-paper")
def evaluate_paper_with_gemini():
    """Grade a complete answer-sheet image/PDF using Gemini Vision (not OCR similarity)."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return jsonify({"error": "Gemini API key is not configured. Add GEMINI_API_KEY to your .env file and restart the app."}), 503
    if genai is None:
        return jsonify({"error": "Gemini SDK is missing. Install requirements with: pip install -r requirements.txt"}), 503

    question_paper = request.files.get("question_paper")
    answer_sheet = request.files.get("answer_sheet")
    if not question_paper or not question_paper.filename:
        return jsonify({"error": "Upload the question paper image or PDF."}), 400
    if not answer_sheet or not answer_sheet.filename:
        return jsonify({"error": "Upload the student's answer sheet image or PDF."}), 400
    if not allowed_file(question_paper.filename) or not allowed_file(answer_sheet.filename):
        return jsonify({"error": "Supported files: JPG, PNG, WEBP, BMP, TIFF, PDF."}), 400

    # Load the saved answer key(s), or accept teacher-provided key text.
    key_id = request.form.get("answer_key_id", "").strip()
    with db() as con:
        if key_id:
            rows = con.execute("SELECT * FROM answer_keys WHERE id=?", (key_id,)).fetchall()
        else:
            rows = con.execute("SELECT * FROM answer_keys ORDER BY id DESC").fetchall()
    keys = [{"question": r["question"], "answer": r["master_answer"],
         "max_marks": r["max_marks"], "keywords_or_rubric": r["keywords"]} for r in rows]
    stamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
    q_name = f"{stamp}_question_{secure_filename(question_paper.filename)}"
    a_name = f"{stamp}_answers_{secure_filename(answer_sheet.filename)}"
    q_path, a_path = UPLOAD_DIR / q_name, UPLOAD_DIR / a_name
    question_paper.save(q_path)
    answer_sheet.save(a_path)
    def mime_for(path):
        return "application/pdf" if path.suffix.lower()==".pdf" else {
            ".jpg":"image/jpeg",".jpeg":"image/jpeg",".png":"image/png",
            ".webp":"image/webp",".bmp":"image/bmp",".tif":"image/tiff",".tiff":"image/tiff"
        }.get(path.suffix.lower(), "image/jpeg")
    try:
        
        prompt = f"""
You are a STRICT university examiner grading handwritten examination answers.

DOCUMENTS:
1. QUESTION PAPER
2. STUDENT ANSWER SHEET
3. OPTIONAL SAVED MARKING SCHEMES

GRADING RULES:

1. Read the question paper carefully and extract every question
   and its maximum printed marks.

2. Grade each answer independently according to the exact
   requirements of that question.

3. Award marks ONLY for relevant, correct, demonstrated knowledge.

4. Completely irrelevant answers must receive ZERO marks,
   even if they are technically correct or well-written.

5. Award partial marks when an answer is partially correct.

6. Missing essential components must lose their corresponding
   marks. Do not assume the student understands something
   that they have not demonstrated.

7. For programming questions, evaluate:
   - Correctness of logic
   - Required syntax or structure
   - Necessary implementation steps
   - Completeness of the requested solution

8. Do not award full marks merely because keywords,
   headings, or some correct steps are present.

9. Create a marking rubric for every question before grading.
   Divide the maximum marks into meaningful criteria.
   Award marks criterion by criterion.

10. Explain exactly which criteria earned marks and which
    criteria lost marks.

11. Saved marking schemes are supplementary guidance.
    Accept equivalent correct answers and alternative
    valid approaches.

12. Do not invent answers, code, explanations, or steps
    that are absent from the student's response.

13. Never exceed the maximum marks of a question.

14. Clearly blank answers receive zero marks.
    Illegible or uncertain answers require teacher review.

15. Calculate total marks by summing individual question
    marks. Do not estimate the total independently.

Q4 EXAMPLE OF STRICT GRADING:
If a question asks for a linked list definition, C node
structure, and insertion at the beginning:
- Correct definition earns its allocated marks.
- Correct node structure earns its allocated marks.
- Correct pointer updates earn their allocated marks.
- Missing essential implementation details lose marks
  according to the rubric.
- Do not automatically award full marks for incomplete code.

OUTPUT:
Return valid JSON only, matching this structure:

{{
  "student_name": "",
  "roll_number": "",
  "answers": [
    {{
      "question": "",
      "question_number": "",
      "student_answer_transcription": "",
      "max_marks": 0,
      "awarded_marks": 0,
      "percentage": 0,
      "feedback": "",
      "credited_points": [],
      "missed_points": [],
      "confidence": "high",
      "review_required": false
    }}
  ],
  "total_awarded": 0,
  "total_max": 0,
  "overall_percentage": 0,
  "overall_feedback": "",
  "review_required": false,
  "review_notes": []
}}

MARKING SCHEMES:
{json.dumps(keys, ensure_ascii=False)}
"""

        with open(q_path, "rb") as fh:
            question_media = genai_types.Part.from_bytes(data=fh.read(), mime_type=mime_for(q_path))
        with open(a_path, "rb") as fh:
            answer_media = genai_types.Part.from_bytes(data=fh.read(), mime_type=mime_for(a_path))
        client = genai.Client(api_key=api_key)
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        response = client.models.generate_content(
            model=model,
            contents=[prompt, "QUESTION PAPER:", question_media, "STUDENT ANSWER SHEET:", answer_media],
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
                max_output_tokens=8192
            )
        )
        result = json.loads(response.text)
        # Guard against malformed/over-limit scores before returning.
        for item in result.get("answers", []):
            maximum = float(item.get("max_marks") or 0)
            score = float(item.get("awarded_marks") or 0)
            if maximum <= 0 or score < 0 or score > maximum:
                item["review_required"] = True
                item["awarded_marks"] = min(max(score, 0), max(maximum, 0))
                item["feedback"] = (item.get("feedback", "") + " Score bounds were checked; teacher review required.").strip()
        result["question_paper_filename"] = q_name
        result["answer_sheet_filename"] = a_name
        result["grading_engine"] = "Gemini Vision"
        result["teacher_review_required"] = True

        # Save one summary row to the shared evaluations history so it appears
        # alongside OCR evaluations. Keep the complete question-wise Gemini
        # response separately for later review; Gemini does not produce OCR
        # keyword/similarity metrics, so those two legacy columns are stored as 0.
        student_name = clean_text(result.get("student_name", "")) or "Unknown student"
        roll_number = clean_text(result.get("roll_number", ""))
        answers = result.get("answers", []) or []
        total_max = float(result.get("total_max") or sum(float(a.get("max_marks") or 0) for a in answers))
        total_awarded = float(result.get("total_awarded") or sum(float(a.get("awarded_marks") or 0) for a in answers))
        percentage = float(result.get("overall_percentage") or
                           ((total_awarded / total_max) * 100 if total_max > 0 else 0))
        percentage = round(max(0, min(100, percentage)), 1)
        total_awarded = round(max(0, min(total_max, total_awarded)), 2)
        grade = ("A" if percentage >= 85 else "B" if percentage >= 70 else
                 "C" if percentage >= 55 else "D" if percentage >= 40 else "Needs Review")
        question_summary = f"Complete paper ({len(answers)} questions)"
        student_transcription = "\\n\\n".join(
            f"{a.get('question_number', 'Question')}: {a.get('student_answer_transcription', '')}"
            for a in answers
        )
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with db() as con:
            cur = con.execute("""
                INSERT INTO evaluations (
                    student_name, roll_number, question, student_answer, master_answer,
                    max_marks, keyword_score, similarity_score, final_score, percentage,
                    grade, feedback, source, image_filename, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                student_name, roll_number, question_summary, student_transcription,
                clean_text(result.get("overall_feedback", "")) or "Gemini Vision paper evaluation",
                total_max, 0, 0, total_awarded, percentage, grade,
                clean_text(result.get("overall_feedback", "")) or "Teacher review required.",
                "Gemini Vision", f"{q_name}; {a_name}", now
            ))
            evaluation_id = cur.lastrowid
            con.execute(
                "INSERT INTO gemini_paper_details (evaluation_id, result_json) VALUES (?, ?)",
                (evaluation_id, json.dumps(result, ensure_ascii=False))
            )
        result["evaluation_id"] = evaluation_id
        result["created_at"] = now
        return jsonify(result)
    except json.JSONDecodeError:
        return jsonify({"error": "Gemini returned an unreadable result. Try a clearer scan or fewer pages."}), 502
    except Exception as exc:
        app.logger.exception("Gemini paper evaluation failed")
        return jsonify({"error": f"Gemini evaluation failed: {str(exc)}"}), 502

@app.get("/api/evaluations")
def evaluations():
    limit = min(int(request.args.get("limit", 50)), 200)
    with db() as con:
        rows = con.execute(
            "SELECT * FROM evaluations ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.get("/api/evaluations/<int:evaluation_id>")
def evaluation(evaluation_id):
    with db() as con:
        row = con.execute("SELECT * FROM evaluations WHERE id=?", (evaluation_id,)).fetchone()
    if not row:
        return jsonify({"error": "Evaluation not found."}), 404
    payload = dict(row)
    if payload.get("source") == "Gemini Vision":
        with db() as con:
            detail = con.execute(
                "SELECT result_json FROM gemini_paper_details WHERE evaluation_id=?",
                (evaluation_id,)
            ).fetchone()
        if detail:
            payload["gemini_details"] = json.loads(detail["result_json"])
    return jsonify(payload)


@app.delete("/api/evaluations/<int:evaluation_id>")
def delete_evaluation(evaluation_id):
    with db() as con:
        cur = con.execute("DELETE FROM evaluations WHERE id=?", (evaluation_id,))
        if cur.rowcount == 0:
            return jsonify({"error": "Evaluation not found."}), 404
    return jsonify({"message": "Evaluation deleted."})


@app.get("/api/dashboard")
def dashboard():
    with db() as con:
        total = con.execute("SELECT COUNT(*) c FROM evaluations").fetchone()["c"]
        students = con.execute("SELECT COUNT(DISTINCT student_name) c FROM evaluations").fetchone()["c"]
        avg = con.execute("SELECT AVG(percentage) a FROM evaluations").fetchone()["a"] or 0
        marks = con.execute("SELECT SUM(final_score) s FROM evaluations").fetchone()["s"] or 0
    return jsonify({
        "total_evaluations": total,
        "total_students": students,
        "average_accuracy": round(avg, 1),
        "total_marks": round(marks, 2)
    })


@app.get("/api/answer-keys")
def answer_keys():
    with db() as con:
        rows = con.execute("SELECT * FROM answer_keys ORDER BY id DESC").fetchall()
    return jsonify([dict(r) for r in rows])


@app.post("/api/answer-keys")
def create_answer_key():
    data = request.get_json(silent=True) or request.form
    title = clean_text(data.get("title", "")) or "Untitled Answer Key"
    question = clean_text(data.get("question", ""))
    master = clean_text(data.get("master_answer", ""))
    keywords = clean_text(data.get("keywords", ""))
    try:
        max_marks = float(data.get("max_marks", 10))
    except Exception:
        max_marks = 10
    if not question or not master:
        return jsonify({"error": "Question and master answer are required."}), 400

    with db() as con:
        cur = con.execute("""
            INSERT INTO answer_keys(title, question, master_answer, keywords, max_marks, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (title, question, master, keywords, max_marks,
              datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    return jsonify({"id": cur.lastrowid, "message": "Answer key saved."})


@app.get("/uploads/<path:filename>")
def uploaded(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    prepare_storage()
    init_db()
    app.run(host="127.0.0.1", port=5000, debug=True)
