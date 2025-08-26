import streamlit as st
import joblib
import docx2txt
import PyPDF2

# ---------------------------
# Load model and vectorizer
# ---------------------------
model = joblib.load("resume_svm_model.pkl")
vectorizer = joblib.load("resume_vectorizer.pkl")

# ---------------------------
# Helper functions
# ---------------------------
def extract_text_from_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    return docx2txt.process(file)

def predict_category(resume_text):
    X = vectorizer.transform([resume_text])
    prediction = model.predict(X)
    return prediction[0]

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Smart Resume Analyzer", page_icon="📄", layout="centered")

st.title("📄 Smart Resume Analyzer")
st.write("Upload a resume (PDF/DOCX) and the app will predict its job category.")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type == "pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    elif file_type == "docx":
        resume_text = extract_text_from_docx(uploaded_file)
    else:
        st.error("❌ Unsupported file type.")
        resume_text = None

    if resume_text:
        st.subheader("📑 Extracted Resume Text:")
        st.text_area("", resume_text[:2000], height=200)  # show first 2000 chars

        if st.button("🔍 Predict Category"):
            category = predict_category(resume_text)
            st.success(f"✅ Predicted Category: **{category}**")
