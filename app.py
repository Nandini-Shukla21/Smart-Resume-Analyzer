import streamlit as st
import joblib
import docx2txt
import PyPDF2
import re
import psycopg2

# ======================
# Load Model & Vectorizer
# ======================
model = joblib.load("resume_svm_model.pkl")
vectorizer = joblib.load("resume_vectorizer.pkl")

# ======================
# DB Connection Function
# ======================
def insert_into_db(resume_text, skills, education, experience, predicted_role, fit_status):
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="resume_db",
            user="postgres",
            password="riya,123"
        )
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO resumes (resume_text, skills, education, experience, predicted_role, fit_status)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (resume_text, ','.join(skills), ','.join(education), str(experience), predicted_role, fit_status))

        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"Database Error: {e}")

# ======================
# Resume Parsing Helpers
# ======================
def extract_text_from_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    return docx2txt.process(file)

def extract_skills(text):
    skills = ["python", "java", "sql", "c++", "machine learning", "deep learning",
              "excel", "aws", "django", "flask", "html", "css", "javascript"]
    found = [s for s in skills if s.lower() in text.lower()]
    return list(set(found))

def extract_education(text):
    education_keywords = ["bachelor", "master", "btech", "mtech", "phd", "mba", "b.sc", "m.sc"]
    matches = [edu for edu in education_keywords if re.search(edu, text.lower())]
    return matches

def extract_experience(text):
    exp_match = re.findall(r'(\d+)\+?\s*(years|year)', text.lower())
    return exp_match if exp_match else ["Not Mentioned"]

def classify_resume(resume_text, job_role):
    X = vectorizer.transform([resume_text])
    predicted_role = model.predict(X)[0]
    fit_status = "Good Fit" if predicted_role.lower() == job_role.lower() else "Not Fit"
    return predicted_role, fit_status

# ======================
# Streamlit App
# ======================
st.title("📄 Smart Resume Analyzer with PostgreSQL")

job_role = st.selectbox("Select Job Role", ["HR", "Data Scientist", "Software Engineer", "Web Developer", "Java Developer", "Python Developer"])

uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file is not None:
    # Extract Text
    if uploaded_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    else:
        resume_text = extract_text_from_docx(uploaded_file)

    if resume_text:
        st.subheader("Extracted Resume Text")
        st.text_area("Resume Content", resume_text[:2000], height=200)

        # Extract details
        skills = extract_skills(resume_text)
        education = extract_education(resume_text)
        experience = extract_experience(resume_text)

        # Predict Fit
        predicted_role, fit_status = classify_resume(resume_text, job_role)

        # Show Results
        st.subheader("🔍 Analysis Results")
        st.write("**Extracted Skills:**", ", ".join(skills) if skills else "Not Found")
        st.write("**Education:**", ", ".join(education) if education else "Not Found")
        st.write("**Experience:**", experience)
        st.write("**Predicted Role:**", predicted_role)
        st.write("**Fit Status:**", fit_status)

        # Save to DB
        if st.button("Save to Database"):
            insert_into_db(resume_text, skills, education, experience, predicted_role, fit_status)
            st.success("✅ Resume saved to database!")
