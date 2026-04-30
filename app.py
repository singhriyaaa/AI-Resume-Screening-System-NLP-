import streamlit as st
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# FUNCTIONS (CORE LOGIC)


def extract_text_from_pdf(file):
    text = ""
    pdf = PyPDF2.PdfReader(file)

    for page in pdf.pages:
        text += page.extract_text()

    return text


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)   # remove special characters
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    return text


def vectorize(resume, job_desc):
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform([resume, job_desc])
    return vectors


def get_similarity(vectors):
    score = cosine_similarity(vectors[0:1], vectors[1:2])
    return score[0][0]

#  STREAMLIT UI


st.title("AI Resume Intelligence System 💼")

st.write("Upload your resume and compare with job description")

uploaded_file = st.file_uploader("Upload Resume (PDF)")
job_desc = st.text_area("Enter Job Description")

# MAIN LOGIC

if uploaded_file and job_desc:

    # Step 1: Extract text
    resume_text = extract_text_from_pdf(uploaded_file)

    # Step 2: Clean text
    cleaned_resume = clean_text(resume_text)

    # Step 3: Convert to vectors
    vectors = vectorize(cleaned_resume, job_desc)

    # Step 4: Calculate similarity
    score = get_similarity(vectors)

    # Step 5: Show result
    st.subheader("Result")
    st.write(f"Match Score: {score*100:.2f}%")
