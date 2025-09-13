from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import joblib
import re
import tempfile
import os

app = Flask(__name__)
CORS(app)

# Load ML model and vectorizer
model = joblib.load("resume_svm_model.pkl")
vectorizer = joblib.load("resume_vectorizer.pkl")

# In-memory store
resume_store = {}

# 🔹 Extract resume text
def extract_text_from_file(file):
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
    job_description = request.form.get('job_description', "")

    if not file:
        return jsonify({"error": "No resume file uploaded"}), 400

    try:
        resume_text = extract_text_from_file(file)
    except Exception as e:
        return jsonify({"error": f"Failed to parse resume: {str(e)}"}), 400

    if not resume_text.strip():
        return jsonify({"error": "Empty resume text"}), 400

    # ML prediction
    X = vectorizer.transform([resume_text])
    prediction = model.predict(X)[0]  # string label (e.g., "DIGITAL-MEDIA")

    # Keyword matching
    resume_keywords = extract_keywords(resume_text)
    job_keywords = extract_keywords(job_description) if job_description else set()

    matched = list(resume_keywords & job_keywords)
    missing = list(job_keywords - resume_keywords)

    resume_id = str(uuid.uuid4())
    resume_store[resume_id] = {
        "resume_text": resume_text,
        "prediction": prediction,   # ✅ keep string instead of int
        "job_description": job_description,
        "resume_keywords": list(resume_keywords),
        "matched_keywords": matched,
        "missing_keywords": missing
    }

    return jsonify({"resume_id": resume_id, "prediction": prediction})

@app.route('/score/<resume_id>', methods=['GET'])
def get_resume_score(resume_id):
    data = resume_store.get(resume_id)
    if not data:
        return jsonify({"error": "Resume not found"}), 404

    total_keywords = len(set(data["resume_keywords"]))
    matched = len(data["matched_keywords"])
    ats_score = int((matched / (total_keywords + 1)) * 100)

    grammar_score = 90
    overall_score = (ats_score + grammar_score) // 2

    return jsonify({
        "ats_score": ats_score,
        "grammar_score": grammar_score,
        "overall_score": overall_score
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
        "job_description": data["job_description"]
    })

@app.route('/grammar/<resume_id>', methods=['GET'])
def get_grammar(resume_id):
    data = resume_store.get(resume_id)
    if not data:
        return jsonify({"error": "Resume not found"}), 404

    return jsonify({
        "score": 90,
        "suggestions": []
    })

@app.route('/compare/<resume_id>', methods=['POST'])
def compare_resume(resume_id):
    job_desc = request.json.get("job_description", "")
    data = resume_store.get(resume_id)
    if not data:
        return jsonify({"error": "Resume not found"}), 404

    jd_keywords = extract_keywords(job_desc)
    match_percentage = int((len(jd_keywords & set(data["resume_keywords"])) / (len(jd_keywords) + 1)) * 100)

    return jsonify({
        "resume_id": resume_id,
        "job_description": job_desc,
        "match_percentage": match_percentage
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "Backend is running 🚀"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
