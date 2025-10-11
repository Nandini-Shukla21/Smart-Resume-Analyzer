# 🌟 Smart-Resume-Analyzer

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python\&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Framework-black?logo=flask)
![React](https://img.shields.io/badge/React-Frontend-61DAFB?logo=react\&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-336791?logo=postgresql\&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-Design-38B2AC?logo=tailwindcss\&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 🧠 Overview

**Smart-Resume-Analyzer** is an **AI-powered resume screening system** designed to help **recruiters and HR professionals** automate the candidate shortlisting process.
It uses **Machine Learning** and **Natural Language Processing (NLP)** to extract and analyze key resume details like **skills, education, and experience**, and then predicts how well a candidate fits a particular job role.

All results are stored in a **PostgreSQL database**, and a modern **React + TailwindCSS dashboard** enables HRs to **review** candidates easily.

---

## ✨ Key Features

✅ **Automated Resume Parsing** – Extracts education, skills, and experience from resumes
🤖 **AI-Based Fit Prediction** – Classifies candidates based on job relevance
📂 **PostgreSQL Integration** – Stores all resumes and predictions securely
🎨 **Modern Dashboard** – Built with React, Vite, and TailwindCSS for a sleek UI
📊 **Data Insights** – Quickly review candidate performance and suitability

---

## 🧰 Tech Stack

| Layer                 | Technologies                                       |
| --------------------- | -------------------------------------------------- |
| **Frontend**          | ⚛️ React · 🧩 TypeScript · 🎨 TailwindCSS · ⚡ Vite |
| **Backend**           | 🐍 Flask · Python 3.10+                            |
| **Database**          | 🐘 PostgreSQL                                      |
| **Machine Learning**  | 🤖 Scikit-learn (SVM) · TF-IDF Vectorizer          |
| **Tools & Utilities** | 🪄 Joblib · 📓 Jupyter Notebook · 🌐 Flask-CORS    |

---

## 📁 Project Structure

```
SMART-RESUME-ANALYZER/
│
├─ frontend/                     # React + Tailwind + Vite frontend
│   ├─ src/                      # React components and pages
│   ├─ public/                   # Static assets
│   ├─ package.json              # Frontend dependencies
│   ├─ vite.config.ts            # Vite configuration
│   └─ tailwind.config.ts        # Tailwind setup
│
├─ app.py                        # Flask backend API
├─ job_keywords.json             # Job role keyword mapping
├─ resume_svm_model.pkl          # ML model for job-fit prediction
├─ resume_vectorizer.pkl         # TF-IDF vectorizer for text features
├─ Resume.csv                    # Sample resume dataset
├─ Resume Parsing.ipynb          # Jupyter notebook for model training
├─ data/                         # Raw data directory
├─ venv/                         # Python virtual environment
└─ README.md                     # Project documentation
```

---

## ⚙️ Installation & Setup

### 🔹 1. Clone the Repository

```bash
git clone https://github.com/Nandini-Shukla21/Smart-Resume-Analyzer.git
cd Smart-Resume-Analyzer
```

### 🔹 2. Backend Setup (Flask)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 🔹 3. Frontend Setup (React)

```bash
cd frontend
npm install
npm run dev
```

### 🔹 4. Database Setup (PostgreSQL)

* Create a database named `resume_analyzer`
* Update credentials in `app.py` or `.env.local`

### 🔹 5. Run Backend Server

```bash
python app.py
```

### 🔹 6. Open the Dashboard

Visit 👉 [http://localhost:3000](http://localhost:8080)

---

## 🧠 How It Works

1. 📤 Upload a resume via the dashboard or API
2. 🧾 The system extracts text and structured information
3. 🤖 ML model predicts whether the candidate fits the job description
4. 💾 Results are stored in PostgreSQL
5. 📊 HRs can review resume of candidates easily from the dashboard

---

## 🌈 Screenshots

*(You can add your real screenshots later!)*

```markdown
![Dashboard Screenshot](screenshots/dashboard.png)
![Upload Page](screenshots/upload.png)
![Results View](screenshots/results.png)
```

---

## 🚀 Future Enhancements

🔸 Integration with LinkedIn for profile import
🔸 Resume ranking system based on match percentage
🔸 Email alerts for shortlisted candidates
🔸 Deep Learning upgrade using BERT/NLP Transformers
🔸 Analytics dashboard for recruitment insights

---

## 🤝 Contributing

We ❤️ contributions!

1. Fork the repository
2. Create a feature branch:

   ```bash
   git checkout -b feature/YourFeature
   ```
3. Commit your changes:

   ```bash
   git commit -m "Added new feature"
   ```
4. Push and create a Pull Request 🚀

---

## 🪪 License

📄 This project is licensed under the **MIT License** — feel free to use, modify, and share with attribution.

---

## 💌 Contact

👩‍💻 **Author:**  Nandini Shukla

📧 **Email:** [nandinishukla023@gmail.com]

🌐 **GitHub:** [https://github.com/Nandini-Shukla21]

💼 **LinkedIn:** [https://www.linkedin.com/in/nandinishukla023/]
---

## 🌟 Support

If you like this project, give it a ⭐ on GitHub!
Your support helps keep the project growing 💖

