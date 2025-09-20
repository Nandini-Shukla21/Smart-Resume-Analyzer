from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import joblib
import re
import tempfile
import os
import json
import logging

app = Flask(__name__)
CORS(app)

# Setup logging to console for easier debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("resume-analyzer")

# --------------------------
# Load ML model & vectorizer
# --------------------------
model = None
vectorizer = None
try:
    model = joblib.load("resume_svm_model.pkl")
    vectorizer = joblib.load("resume_vectorizer.pkl")
    logger.info("Loaded model and vectorizer.")
except Exception as e:
    logger.warning("Failed to load ML model/vectorizer: %s", e)
    # We'll still allow the rest of the API to work; prediction will be "unknown".

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

# Build lowercase lookup for job roles -> canonical key
VALID_ROLES = {role.lower(): role for role in JOB_KEYWORDS.keys()}

# --------------------------
# Grammar tool (GingerIt) - optional
# --------------------------
try:
    from gingerit.gingerit import GingerIt
    ginger = GingerIt()
    ginger_available = True
    logger.info("GingerIt grammar checker available.")
except Exception as e:
    ginger = None
    ginger_available = False
    logger.warning("GingerIt not available (will skip grammar): %s", e)

# --------------------------
# In-memory store
# --------------------------
resume_store = {}

# --------------------------
# Utility functions
# --------------------------
def extract_text_from_file(file) -> str:
    """
    Save uploaded file to a temp file then extract text depending on extension.
    Returns extracted text (may be empty).
    """
    resume_text = ""
    ext = os.path.splitext(file.filename)[1].lower()
    # Save file temporarily
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
            # txt or other - attempt to read as text
            with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                resume_text = f.read()
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return resume_text

def extract_keywords(text):
    """
    Very simple keyword extractor - returns set of lowercase words of length >= 3.
    (You can replace with more sophisticated NLP later.)
    """
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return set(words)

def canonicalize_job_role(incoming_role: str):
    """
    Accepts incoming_role and returns the canonical role name from JOB_KEYWORDS keys
    (preserves original casing). Returns None if not found.
    """
    if not incoming_role:
        return None
    key = incoming_role.strip().lower()
    return VALID_ROLES.get(key)

# --------------------------
# Endpoints
# --------------------------
@app.route("/roles", methods=["GET"])
def list_roles():
    """Return available job roles (for frontend dropdown)."""
    return jsonify({"roles": list(JOB_KEYWORDS.keys())})

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    # Accept both names: frontends may send 'job_role' or 'job_description' (older code)
    file = request.files.get('resume')
    job_role_form = request.form.get('job_role')
    job_description_form = request.form.get('job_description')

    # Prefer explicit job_role param, fallback to job_description (legacy)
    incoming_role = job_role_form or job_description_form or ""
    incoming_role = incoming_role.strip()

    logger.info("Received /analyze request. Form keys: %s. job_role='%s'", dict(request.form), incoming_role)

    if not file:
        return jsonify({"error": "No resume file uploaded"}), 400

    # Canonicalize role using JOB_KEYWORDS; if not found return helpful error
    canonical_role = canonicalize_job_role(incoming_role)
    if not canonical_role:
        return jsonify({
            "error": "Invalid or missing job role",
            "provided": incoming_role,
            "available_roles": list(JOB_KEYWORDS.keys())
        }), 400

    # Extract text
    try:
        resume_text = extract_text_from_file(file)
    except Exception as e:
        logger.exception("Failed to extract resume text")
        return jsonify({"error": f"Failed to parse resume: {str(e)}"}), 400

    if not resume_text.strip():
        return jsonify({"error": "Empty resume text after parsing. Make sure this is a machine-readable PDF/DOCX (not an image scan)."}), 400

    # ML prediction (if model available)
    prediction = "unknown"
    try:
        if model is not None and vectorizer is not None:
            X = vectorizer.transform([resume_text])
            pred = model.predict(X)
            if len(pred) > 0:
                prediction = pred[0]
        else:
            logger.warning("Model or vectorizer missing; returning prediction='unknown'.")
    except Exception as e:
        logger.exception("Model prediction failed; returning 'unknown'. Error: %s", e)
        prediction = "unknown"

    # Keyword matching using canonical role keywords
    resume_keywords = extract_keywords(resume_text)  # set of lowercase words
    job_keywords = set([w.lower() for w in JOB_KEYWORDS.get(canonical_role, [])])

    matched = sorted(list(resume_keywords & job_keywords))
    missing = sorted(list(job_keywords - resume_keywords))

    # Grammar analysis (if ginger is available)
    grammar_suggestions = []
    grammar_score = 90
    if ginger_available and ginger is not None:
        try:
            grammar_result = ginger.parse(resume_text)
            # GingerIt returns 'corrections' sometimes under different keys; be defensive
            corrections = grammar_result.get("corrections") or grammar_result.get("matches") or []
            # Normalize into text/suggestion pairs if possible
            grammar_suggestions = []
            for c in corrections:
                # Some gingerit versions return dicts differently; handle both shapes
                if isinstance(c, dict):
                    # fields vary; try common keys
                    txt = c.get("text") or c.get("sentence") or ""
                    corr = c.get("correct") or c.get("replacements") or ""
                    # If replacements is list, join
                    if isinstance(corr, list):
                        corr = ", ".join(map(str, corr))
                    grammar_suggestions.append({"text": txt, "suggestion": corr})
                else:
                    # fallback: stringify
                    grammar_suggestions.append({"text": str(c), "suggestion": ""})
            grammar_score = max(50, 100 - len(grammar_suggestions) * 2)
        except Exception as e:
            logger.exception("Ginger parse failed; skipping grammar suggestions: %s", e)
            grammar_suggestions = []
            grammar_score = 90

    # store
    resume_id = str(uuid.uuid4())
    resume_store[resume_id] = {
        "resume_text": resume_text,
        "prediction": prediction,
        "job_role": canonical_role,
        "resume_keywords": list(resume_keywords),
        "matched_keywords": matched,
        "missing_keywords": missing,
        "grammar_suggestions": grammar_suggestions,
        "grammar_score": grammar_score,
    }

    logger.info("Analysis saved resume_id=%s role=%s matched=%d missing=%d", resume_id, canonical_role, len(matched), len(missing))

    return jsonify({"resume_id": resume_id, "prediction": prediction, "job_role": canonical_role})

@app.route('/score/<resume_id>', methods=['GET'])
def get_resume_score(resume_id):
    data = resume_store.get(resume_id)
    if not data:
        return jsonify({"error": "Resume not found"}), 404

    matched = len(data.get("matched_keywords", []))
    missing = len(data.get("missing_keywords", []))
    total_keywords = matched + missing
    ats_score = int((matched / (total_keywords + 1)) * 100)  # +1 to avoid div-by-zero

    grammar_score = data.get("grammar_score", 90)
    overall_score = int((ats_score + grammar_score) // 2)

    sections_analysis = {
        "experience": {"score": 75, "suggestions": []},
        "skills": {"score": 70, "suggestions": []},
        "education": {"score": 80, "suggestions": []},
        "summary": {"score": 65, "suggestions": []}
    }

    return jsonify({
        "ats_score": ats_score,
        "grammar_score": grammar_score,
        "overall_score": overall_score,
        "sections_analysis": sections_analysis
    })

@app.route('/keywords/<resume_id>', methods=['GET'])
def get_keywords(resume_id):
    """
    Returns the stored matched/missing keywords for the given resume_id.
    Accepts optional ?job_role=... query param to evaluate matched/missing
    vs a different role on the fly.
    """
    data = resume_store.get(resume_id)
    if not data:
        return jsonify({"error": "Resume not found"}), 404

    # optional query param to override/check against a specific job role
    job_role_q = request.args.get("job_role", "").strip()
    if job_role_q:
        canonical = canonicalize_job_role(job_role_q)
        if not canonical:
            return jsonify({
                "error": "Invalid job_role query param",
                "provided": job_role_q,
                "available_roles": list(JOB_KEYWORDS.keys())
            }), 400
        job_keywords = set([w.lower() for w in JOB_KEYWORDS.get(canonical, [])])
        resume_keywords = set(data.get("resume_keywords", []))
        matched = sorted(list(resume_keywords & job_keywords))
        missing = sorted(list(job_keywords - resume_keywords))
        total_keywords = len(job_keywords)
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

    # default: return stored matched/missing (calculated at analyze time)
    total_keywords = len(data.get("resume_keywords", []))
    matched_count = len(data.get("matched_keywords", []))
    density = round(matched_count / (total_keywords + 1), 2)

    return jsonify({
        "total_keywords": total_keywords,
        "matched_keywords": matched_count,
        "matched": data.get("matched_keywords", []),
        "missing": data.get("missing_keywords", []),
        "keyword_density": density,
        "job_role": data.get("job_role")
    })

@app.route('/grammar/<resume_id>', methods=['GET'])
def get_grammar(resume_id):
    data = resume_store.get(resume_id)
    if not data:
        return jsonify({"error": "Resume not found"}), 404

    return jsonify({
        "score": data.get("grammar_score", 90),
        "suggestions": data.get("grammar_suggestions", [])
    })

@app.route('/compare/<resume_id>', methods=['POST'])
def compare_resume(resume_id):
    json_body = request.get_json(silent=True) or {}
    job_role = (json_body.get("job_role") or json_body.get("job_description") or "").strip()
    data = resume_store.get(resume_id)
    if not data:
        return jsonify({"error": "Resume not found"}), 404

    canonical = canonicalize_job_role(job_role)
    if not canonical:
        return jsonify({
            "error": "Invalid job role",
            "provided": job_role,
            "available_roles": list(JOB_KEYWORDS.keys())
        }), 400

    jd_keywords = set(w.lower() for w in JOB_KEYWORDS.get(canonical, []))
    resume_keywords = set(data.get("resume_keywords", []))
    match_percentage = int((len(jd_keywords & resume_keywords) / (len(jd_keywords) + 1)) * 100)

    return jsonify({
        "resume_id": resume_id,
        "job_role": canonical,
        "match_percentage": match_percentage
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "Backend is running 🚀",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "ginger_available": ginger_available
    })

# --------------------------
# Start
# --------------------------
if __name__ == '__main__':
    app.run(debug=True, port=5000)
