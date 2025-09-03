from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import joblib

app = Flask(__name__)
CORS(app)

# Load ML model and vectorizer
model = joblib.load("resume_svm_model.pkl")
vectorizer = joblib.load("resume_vectorizer.pkl")

# In-memory store (can replace with DB later)
resume_store = {}

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    file = request.files.get('resume')
    job_description = request.form.get('job_description')

    if not file:
        return jsonify({"error": "No resume file uploaded"}), 400

    # For now: assume text file (later replace with PDF/DOCX parser)
    try:
        resume_text = file.read().decode("utf-8", errors="ignore")
    except Exception:
        return jsonify({"error": "Failed to read resume"}), 400

    # Transform resume text and predict
    X = vectorizer.transform([resume_text])
    prediction = model.predict(X)[0]

    resume_id = str(uuid.uuid4())
    resume_store[resume_id] = {
        "resume_text": resume_text,
        "ats_score": 82,
        "grammar_score": 90,
        "prediction": int(prediction),
        "job_description": job_description
    }

    return jsonify({"resume_id": resume_id, "prediction": int(prediction)})

@app.route('/score/<resume_id>', methods=['GET'])
def get_resume_score(resume_id):
    data = resume_store.get(resume_id)
    if not data:
        return jsonify({"error": "Resume not found"}), 404
    return jsonify({
        "ats_score": data["ats_score"],
        "grammar_score": data["grammar_score"],
        "overall_score": (data["ats_score"] + data["grammar_score"]) // 2
    })

@app.route('/keywords/<resume_id>', methods=['GET'])
def get_keywords(resume_id):
    job_desc = request.args.get("job_description", "")
    data = resume_store.get(resume_id)
    if not data:
        return jsonify({"error": "Resume not found"}), 404
    
    return jsonify({
        "matched": ["Python", "React"],
        "missing": ["AWS", "Docker"],
        "job_description": job_desc
    })

@app.route('/grammar/<resume_id>', methods=['GET'])
def get_grammar(resume_id):
    data = resume_store.get(resume_id)
    if not data:
        return jsonify({"error": "Resume not found"}), 404
    
    return jsonify({
        "score": data["grammar_score"],
        "suggestions": [
            {"text": "Change 'recieve' to 'receive'"},
            {"text": "Add comma after introductory phrase"}
        ]
    })

@app.route('/compare/<resume_id>', methods=['POST'])
def compare_resume(resume_id):
    job_desc = request.json.get("job_description", "")
    data = resume_store.get(resume_id)
    if not data:
        return jsonify({"error": "Resume not found"}), 404
    
    return jsonify({
        "resume_id": resume_id,
        "job_description": job_desc,
        "match_percentage": 78
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "Backend is running 🚀"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
