from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import joblib
import re
import tempfile
import os
import json
from gingerit.gingerit import GingerIt

app = Flask(__name__)
CORS(app)

# Load ML model and vectorizer
model = joblib.load("resume_svm_model.pkl")
vectorizer = joblib.load("resume_vectorizer.pkl")

# Load job-role keywords JSON
with open("job_keywords.json", "r") as f:
    JOB_KEYWORDS = json.load(f)

# In-memory store
resume_store = {}

# GingerIt for grammar checking
ginger = GingerIt()

# 🔹 Extract resume text
def extract_text_from_file(file):
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
        else:  # txt
            with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                resume_text = f.read()
    finally:
        os.remove(tmp_path)

    return resume_text

# 🔹 Simple keyword extractor
def extract_keywords(text):
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return set(words)

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    file = request.files.get('resume')
    job_role = request.form.get('job_role', "")

    if not file:
        return jsonify({"error": "No resume file uploaded"}), 400
    if not job_role or job_role not in JOB_KEYWORDS:
        return jsonify({"error": "Invalid or missing job role"}), 400

    try:
        resume_text = extract_text_from_file(file)
    except Exception as e:
        return jsonify({"error": f"Failed to parse resume: {str(e)}"}), 400

    if not resume_text.strip():
        return jsonify({"error": "Empty resume text"}), 400

    # ML prediction
    X = vectorizer.transform([resume_text])
    prediction = model.predict(X)[0]

    # Keyword matching (resume vs job role keywords)
    resume_keywords = extract_keywords(resume_text)
    job_keywords = set(JOB_KEYWORDS[job_role])

    matched = list(resume_keywords & job_keywords)
    missing = list(job_keywords - resume_keywords)

    # Grammar analysis with GingerIt
    try:
        grammar_result = ginger.parse(resume_text)
        grammar_suggestions = [
            {"text": item.get("text", ""), "suggestion": item.get("correct", "")}
            for item in grammar_result.get("corrections", [])
        ]
        grammar_score = max(50, 100 - len(grammar_suggestions) * 2)
    except Exception:
        grammar_suggestions = []
        grammar_score = 90

    resume_id = str(uuid.uuid4())
    resume_store[resume_id] = {
        "resume_text": resume_text,
        "prediction": prediction,
        "job_role": job_role,
        "resume_keywords": list(resume_keywords),
        "matched_keywords": matched,
        "missing_keywords": missing,
        "grammar_suggestions": grammar_suggestions,
        "grammar_score": grammar_score,
    }

    return jsonify({"resume_id": resume_id, "prediction": prediction, "job_role": job_role})

@app.route('/score/<resume_id>', methods=['GET'])
def get_resume_score(resume_id):
    data = resume_store.get(resume_id)
    if not data:
        return jsonify({"error": "Resume not found"}), 404

    total_keywords = len(data["resume_keywords"])
    matched = len(data["matched_keywords"])
    ats_score = int((matched / (len(data["missing_keywords"]) + matched + 1)) * 100)

    grammar_score = data.get("grammar_score", 90)
    overall_score = (ats_score + grammar_score) // 2

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
    data = resume_store.get(resume_id)
    if not data:
        return jsonify({"error": "Resume not found"}), 404

    total_keywords = len(data["resume_keywords"])
    matched = len(data["matched_keywords"])

    return jsonify({
        "total_keywords": total_keywords,
        "matched_keywords": matched,
        "matched": data["matched_keywords"],
        "missing": data["missing_keywords"],
        "keyword_density": round(matched / (total_keywords + 1), 2),
        "job_role": data["job_role"]
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
    job_role = request.json.get("job_role", "")
    data = resume_store.get(resume_id)
    if not data:
        return jsonify({"error": "Resume not found"}), 404
    if not job_role or job_role not in JOB_KEYWORDS:
        return jsonify({"error": "Invalid or missing job role"}), 400

    jd_keywords = set(JOB_KEYWORDS[job_role])
    match_percentage = int((len(jd_keywords & set(data["resume_keywords"])) / (len(jd_keywords) + 1)) * 100)

    return jsonify({
        "resume_id": resume_id,
        "job_role": job_role,
        "match_percentage": match_percentage
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "Backend is running 🚀"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
