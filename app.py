# app.py
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
    logger.info("GingerIt available.")
except Exception as e:
    ginger, ginger_available = None, False
    logger.info("GingerIt not available: %s", e)

# --------------------------
# Postgres connection
# --------------------------
def get_db_connection():
    # Update these credentials to match your local Postgres / pgAdmin settings
    return psycopg2.connect(
        dbname="my_db",       # <-- change if needed
        user="postgres",      # <-- change if needed
        password="riya,123",  # <-- change if needed
        host="localhost",
        port="5432",
        cursor_factory=RealDictCursor
    )

# --------------------------
# Utility functions
# --------------------------
def extract_text_from_file(file) -> str:
    """Save uploaded file to temp then extract text depending on extension."""
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
            # try reading as plain text
            with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                resume_text = f.read()
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return resume_text

def extract_name(text: str) -> str:
    """Very simple name extractor: first non-empty line."""
    for line in text.splitlines():
        s = line.strip()
        if s:
            # Return first meaningful line (could be improved with NLP)
            return s
    return "Unknown"

def normalize_text(text: str) -> str:
    """Return normalized lowercase text for substring matching."""
    return text.lower()

def match_job_keywords(resume_text: str, job_keywords_list):
    """
    Match job keywords (single and multi-word phrases) against resume_text.
    Returns (matched_list, missing_list). Matching is case-insensitive substring search.
    """
    resume_norm = normalize_text(resume_text)
    matched = []
    missing = []
    for kw in job_keywords_list:
        kw_normal = kw.strip().lower()
        if not kw_normal:
            continue
        # Use simple substring containment — good for multi-word phrases too
        if kw_normal in resume_norm:
            matched.append(kw)
        else:
            missing.append(kw)
    return matched, missing

# --------------------------
# Endpoints
# --------------------------
@app.route("/roles", methods=["GET"])
def list_roles():
    return jsonify({"roles": list(JOB_KEYWORDS.keys())})

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    # Accept 'job_role' or legacy 'job_description' in form data
    file = request.files.get('resume')
    job_role_form = request.form.get('job_role')
    job_description_form = request.form.get('job_description')

    incoming_role = (job_role_form or job_description_form or "").strip()
    canonical_role = VALID_ROLES.get(incoming_role.lower())

    logger.info("Analyze called. job_role provided: '%s' -> canonical: %s", incoming_role, canonical_role)

    if not file:
        return jsonify({"error": "No resume file uploaded"}), 400
    if not canonical_role:
        return jsonify({
            "error": "Invalid or missing job role",
            "provided": incoming_role,
            "available_roles": list(JOB_KEYWORDS.keys())
        }), 400

    # extract text
    try:
        resume_text = extract_text_from_file(file)
    except Exception as e:
        logger.exception("Failed to parse resume")
        return jsonify({"error": f"Failed to parse resume: {str(e)}"}), 400

    if not resume_text.strip():
        return jsonify({"error": "Empty resume text after parsing. Ensure PDF/DOCX is machine-readable (not scanned image)."}), 400

    # ML prediction (optional)
    prediction = "unknown"
    try:
        if model and vectorizer:
            X = vectorizer.transform([resume_text])
            pred = model.predict(X)
            if len(pred) > 0:
                prediction = pred[0]
    except Exception as e:
        logger.warning("Model prediction failed: %s", e)
        prediction = "unknown"

    # Keyword analysis (multi-word-aware)
    job_keywords_list = JOB_KEYWORDS.get(canonical_role, [])  # keep original casing from JSON
    matched, missing = match_job_keywords(resume_text, job_keywords_list)

    # Grammar check (optional)
    grammar_suggestions = []
    grammar_score = 90
    if ginger_available and ginger:
        try:
            res = ginger.parse(resume_text)
            # gingerit returns corrections / matches in different shapes — be defensive
            corrections = res.get("corrections") or res.get("matches") or []
            grammar_suggestions = []
            for c in corrections:
                if isinstance(c, dict):
                    txt = c.get("text") or c.get("sentence") or ""
                    corr = c.get("correct") or c.get("replacements") or ""
                    if isinstance(corr, list):
                        corr = ", ".join(map(str, corr))
                    grammar_suggestions.append({"text": txt, "suggestion": corr})
                else:
                    grammar_suggestions.append({"text": str(c), "suggestion": ""})
            grammar_score = max(50, 100 - len(grammar_suggestions) * 2)
        except Exception as e:
            logger.warning("Ginger parse failed: %s", e)
            grammar_suggestions = []
            grammar_score = 90

    # Compute simple scores
    resume_keywords_set = set(re.findall(r"\b[a-zA-Z0-9+-]{2,}\b", resume_text.lower()))
    matched_count = len(matched)
    total_job_keywords = len(job_keywords_list)
    ats_score = int((matched_count / (total_job_keywords + 1)) * 100)  # +1 to avoid div0
    overall_score = int((ats_score + grammar_score) // 2)

    # basic qualifications extraction
    qualifications = " | ".join([s for s in resume_keywords_set if s.lower() in {"btech", "mtech", "msc", "bsc", "mba"}])

    # persist to DB
    resume_id = str(uuid.uuid4())
    name = extract_name(resume_text)
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
            ",".join(matched),  # store matched skills as CSV
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
        logger.exception("Failed to save resume to Postgres: %s", e)

    # Return the full shaped data the frontend expects (Home -> Dashboard)
    response = {
        "resume_id": resume_id,
        "prediction": prediction,
        "job_role": canonical_role,
        "overall_score": overall_score,
        "ats_score": ats_score,
        "grammar_score": grammar_score,
        "keyword_match_percentage": int((matched_count / (total_job_keywords + 1)) * 100),
        "sections_analysis": {   # stubbed section scores (frontend expects object)
            "experience": {"score": 70},
            "skills": {"score": 80},
            "education": {"score": 65},
            "summary": {"score": 50}
        },
        "matched_skills": matched,
        "missing_skills": missing,
        "keyword_analysis": {
            "matched_keywords": matched_count,
            "total_keywords": total_job_keywords,
            "keyword_density": round(matched_count / (len(resume_keywords_set) + 1), 3)
        },
        "grammar_suggestions": grammar_suggestions,
        "readability_score": 80
    }
    return jsonify(response)

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
            "ats_score": ats_score,
            "grammar_score": grammar_score,
            "overall_score": overall_score,
            "sections_analysis": {
                "experience": {"score": 70},
                "skills": {"score": 80},
                "education": {"score": 65},
                "summary": {"score": 50}
            }
        })
    except Exception as e:
        logger.exception("Error in get_resume_score")
        return jsonify({"error": str(e)}), 500

@app.route('/keywords/<resume_id>', methods=['GET'])
def get_keywords(resume_id):
    """
    Returns keyword-related data for given resume_id.
    Optional query param: job_role to compute vs different role.
    """
    job_role_q = request.args.get("job_role", "").strip()
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT job_role, skills, resume_text FROM resumes WHERE resume_id=%s", (resume_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row:
            return jsonify({"error": "Resume not found"}), 404

        # Use stored skills if no override role; else compute against provided job_role
        
        if job_role_q:
            canonical = VALID_ROLES.get(job_role_q.lower())
            if not canonical:
                return jsonify({"error": "Invalid job_role query param", "available_roles": list(JOB_KEYWORDS.keys())}), 400
            job_keywords_list = JOB_KEYWORDS.get(canonical, [])
            matched, missing = match_job_keywords(row["resume_text"], job_keywords_list)
            total_keywords = len(job_keywords_list)
            matched_count = len(matched)
            density = round(matched_count / (total_keywords + 1), 2)
            return jsonify({
                "total_keywords": total_keywords,
                "matched_keywords": matched_count,
                "matched": matched,
                "missing": missing,
                "keyword_density": density,
                "job_role": canonical
            })

        # default: use stored skills field (CSV)

        stored_skills = row.get("skills") or ""
        skills_list = stored_skills.split(",") if stored_skills else []
        total_keywords = len(skills_list)
        matched_count = len(skills_list)
        density = round(matched_count / (total_keywords + 1), 2)

        return jsonify({
            "total_keywords": total_keywords,
            "matched_keywords": matched_count,
            "matched": skills_list,
            "missing": [],
            "keyword_density": density,
            "job_role": row.get("job_role")
        })
    except Exception as e:
        logger.exception("Error in get_keywords")
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
            "score": row["grammar_score"],
            "suggestions": []
        })
    except Exception as e:
        logger.exception("Error in get_grammar")
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

