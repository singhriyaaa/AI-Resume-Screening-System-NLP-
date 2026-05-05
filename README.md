# AI Resume Intelligence System

An NLP-based resume screening tool that matches resumes with job descriptions using TF-IDF, cosine similarity, and skill gap analysis. Built with Python and deployed via Streamlit.

---

## Features

- Resume Match Scoring — Hybrid scoring using tuned TF-IDF (bigrams, L2 norm) + keyword overlap for accurate matching
- Skill Gap Analysis — Classifies skills as Matched, Missing, or Extra based on job description requirements
- Multi-Candidate Ranking — Upload multiple resumes and rank all candidates by job relevance in one go
- Visualizations — Match score gauge bar and skill gap summary chart using Matplotlib and Seaborn
- Interactive UI — Two-tab Streamlit web app with real-time results

---

## Tech Stack

- Language: Python 3.11
- NLP: scikit-learn (TF-IDF, cosine similarity)
- Visualization: Matplotlib, Seaborn
- PDF Parsing: PyPDF2
- Web App: Streamlit

---

## Project Structure

```
resume-intelligence-system/
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Setup

1. Clone the repository
```bash
git clone https://github.com/singhriyaaa/resume-intelligence-system.git
cd resume-intelligence-system
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the app
```bash
streamlit run app.py
```

---

## How It Works

The system uses a hybrid scoring approach — 70% weight on TF-IDF cosine similarity and 30% on direct keyword overlap. This reduces over-penalization of shared terms and gives more accurate results compared to basic TF-IDF alone.

For skill gap analysis, the app checks the resume and job description against a list of 30+ common tech skills and classifies each one as matched, missing, or extra. Missing skills are ones the job requires but the resume doesn't have. Extra skills are present in the resume but not relevant to that specific job.

For multi-candidate ranking, the same hybrid scoring runs on each uploaded resume and sorts candidates from highest to lowest match score.

---

## Author

Riya Singh  
GitHub: https://github.com/singhriyaaa  
LinkedIn: https://linkedin.com/in/riya-singh-58b668244  
Email: riyasingh46143@gmail.com

## Output
[AI Resume Intelligence System1.pdf](https://github.com/user-attachments/files/27385315/AI.Resume.Intelligence.System1.pdf)
[AI Resume Intelligence System.pdf](https://github.com/user-attachments/files/27385316/AI.Resume.Intelligence.System.pdf)


