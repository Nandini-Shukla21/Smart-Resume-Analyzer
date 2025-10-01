from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import joblib
import re
import tempfile
import os
import json
import logging
import psycopg2
from psycopg2.extras import RealDictCursor

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("resume-analyzer")

# --------------------------
# Load ML model & vectorizer
# --------------------------
model, vectorizer = None, None
try:
    model = joblib.load("resume_svm_model.pkl")
    vectorizer = joblib.load("resume_vectorizer.pkl")
    logger.info("Loaded model and vectorizer.")
except Exception as e:
    logger.warning("Failed to load ML model/vectorizer: %s", e)

# --------------------------
# Load job-role keywords JSON
# --------------------------
JOB_KEYWORDS = {}
try:
    with open("job_keywords.json", "r", encoding="utf-8") as f:
        JOB_KEYWORDS = json.load(f)
    logger.info("Loaded job_keywords.json with %d roles.", len(JOB_KEYWORDS))
except Exception as e:
    logger.error("Failed to load job_keywords.json: %s", e)
    JOB_KEYWORDS = {}

VALID_ROLES = {role.lower(): role for role in JOB_KEYWORDS.keys()}

# --------------------------
# Grammar tool (optional)
# --------------------------
try:
    from gingerit.gingerit import GingerIt
    ginger = GingerIt()
    ginger_available = True
except Exception:
    ginger, ginger_available = None, False

# --------------------------
# Postgres connection
# --------------------------
def get_db_connection():
    return psycopg2.connect(
        dbname="my_db",
        user="postgres",
        password="riya,123",
        host="localhost",
        port="5432",
        cursor_factory=RealDictCursor
    )

# --------------------------
# Utility functions
# --------------------------
def extract_text_from_file(file) -> str:
    resume_text = ""
    ext = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        if ext == ".pdf":
            import pdfplumber
            with pdfplumber.open(tmp_path) as pdf:
                for page in pdf.pages:
                    resume_text += page.extract_text() or ""
        elif ext == ".docx":
            import docx
            doc = docx.Document(tmp_path)
            for para in doc.paragraphs:
                resume_text += para.text + "\n"
        else:
            with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                resume_text = f.read()
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return resume_text


def extract_keywords(text):
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return set(words)


def canonicalize_job_role(incoming_role: str):
    if not incoming_role:
        return None
    return VALID_ROLES.get(incoming_role.strip().lower())


def extract_name(text):
    lines = text.splitlines()
    if lines:
        return lines[0].strip()
    return "Unknown"

# --------------------------
# Endpoints
# --------------------------
@app.route("/roles", methods=["GET"])
def list_roles():
    return jsonify({"roles": list(JOB_KEYWORDS.keys())})


@app.route('/analyze', methods=['POST'])
def analyze_resume():
    file = request.files.get('resume')
    job_role_form = request.form.get('job_role')
    job_description_form = request.form.get('job_description')

    incoming_role = (job_role_form or job_description_form or "").strip()
    canonical_role = canonicalize_job_role(incoming_role)

    if not file:
        return jsonify({"error": "No resume file uploaded"}), 400
    if not canonical_role:
        return jsonify({
            "error": "Invalid or missing job role",
            "provided": incoming_role,
            "available_roles": list(JOB_KEYWORDS.keys())
        }), 400

    try:
        resume_text = extract_text_from_file(file)
    except Exception as e:
        return jsonify({"error": f"Failed to parse resume: {str(e)}"}), 400

    if not resume_text.strip():
        return jsonify({"error": "Empty resume text after parsing"}), 400

    # ML prediction
    prediction = "unknown"
    try:
        if model and vectorizer:
            X = vectorizer.transform([resume_text])
            pred = model.predict(X)
            if len(pred) > 0:
                prediction = pred[0]
    except Exception as e:
        logger.warning("Model prediction failed: %s", e)

    # Keyword analysis
    resume_keywords = extract_keywords(resume_text)
    job_keywords = set([w.lower() for w in JOB_KEYWORDS.get(canonical_role, [])])
    matched = sorted(list(resume_keywords & job_keywords))
    missing = sorted(list(job_keywords - resume_keywords))

    # Grammar (safe fallback)
    grammar_suggestions = []
    grammar_score = 90
    if ginger_available and ginger:
        try:
            grammar_result = ginger.parse(resume_text)
            corrections = grammar_result.get("corrections") or []
            for c in corrections:
                grammar_suggestions.append({
                    "text": c.get("text", ""),
                    "suggestion": c.get("correct", "")
                })
            grammar_score = max(50, 100 - len(grammar_suggestions) * 2)
        except Exception as e:
            logger.warning("Grammar check skipped: %s", e)

    # Save to Postgres
    resume_id = str(uuid.uuid4())
    name = extract_name(resume_text)
    ats_score = int((len(matched) / (len(matched) + len(missing) + 1)) * 100)
    overall_score = int((ats_score + grammar_score) // 2)
    qualifications = " | ".join([s for s in resume_keywords if s.lower() in ["btech", "mtech", "msc", "bsc", "mba"]])

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO resumes 
            (resume_id, name, job_role, skills, ats_score, grammar_score, qualifications, resume_text, prediction)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (resume_id) DO NOTHING
        """, (
            resume_id,
            name,
            canonical_role,
            ",".join(matched),
            ats_score,
            grammar_score,
            qualifications,
            resume_text,
            prediction
        ))
        conn.commit()
        cur.close()
        conn.close()
        logger.info("Saved resume_id=%s to Postgres", resume_id)
    except Exception as e:
        logger.error("Failed to save resume: %s", e)

    # ✅ return full JSON for dashboard
    return jsonify({
        "resume_id": resume_id,
        "prediction": prediction,
        "job_role": canonical_role,
        "overall_score": overall_score,
        "ats_score": ats_score,
        "grammar_score": grammar_score,
        "keyword_match_percentage": int((len(matched) / (len(job_keywords) + 1)) * 100),
        "sections_analysis": {
            "experience": {"score": 70},   # stubbed
            "skills": {"score": 80},       # stubbed
            "education": {"score": 65},    # stubbed
            "summary": {"score": 50}       # stubbed
        },
        "matched_skills": matched,
        "missing_skills": missing,
        "keyword_analysis": {
            "matched_keywords": len(matched),
            "total_keywords": len(job_keywords),
            "keyword_density": round(len(matched) / (len(resume_keywords) + 1), 3)
        },
        "grammar_suggestions": grammar_suggestions,
        "readability_score": 80
    })


@app.route('/score/<resume_id>', methods=['GET'])
def get_resume_score(resume_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT ats_score, grammar_score FROM resumes WHERE resume_id=%s", (resume_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            return jsonify({"error": "Resume not found"}), 404

        ats_score = row["ats_score"]
        grammar_score = row["grammar_score"]
        overall_score = int((ats_score + grammar_score) // 2)

        return jsonify({
            "resume_id": resume_id,
            "ats_score": ats_score,
            "grammar_score": grammar_score,
            "overall_score": overall_score
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/keywords/<resume_id>', methods=['GET'])
def get_keywords(resume_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT job_role, skills FROM resumes WHERE resume_id=%s", (resume_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            return jsonify({"error": "Resume not found"}), 404

        skills = row["skills"].split(",") if row["skills"] else []
        return jsonify({
            "job_role": row["job_role"],
            "matched_skills": skills,
            "missing_skills": [],
            "keyword_analysis": {
                "total_keywords": len(skills),
                "matched_keywords": len(skills)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/grammar/<resume_id>', methods=['GET'])
def get_grammar(resume_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT grammar_score FROM resumes WHERE resume_id=%s", (resume_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            return jsonify({"error": "Resume not found"}), 404

        return jsonify({
            "grammar_score": row["grammar_score"],
            "suggestions": []
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "Backend is running 🚀",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "ginger_available": ginger_available
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
