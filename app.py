import streamlit as st
import PyPDF2
import re
import matplotlib.pyplot as plt
import seaborn as sns
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


#  Vectorize (Convert Text to Numbers)
# tokenization is done inside TfidfVectorizer
# feature engineering: ngram_range + max_features tuning

def vectorize(resume, job_desc):
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words='english',
        use_idf=False,   # FIX: stops penalizing shared terms
        norm='l2'
    )
    vectors = tfidf.fit_transform([resume, job_desc])
    return vectors, tfidf

# Get Similarity Score


def get_similarity(vectors):
    score = cosine_similarity(vectors[0:1], vectors[1:2])
    return round(score[0][0] * 100, 2)


def get_hybrid_score(resume_text, job_desc_text):
    cleaned_resume = clean_text(resume_text)
    cleaned_job = clean_text(job_desc_text)

    vectors, _ = vectorize(cleaned_resume, cleaned_job)
    tf_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100

    job_words = {w for w in cleaned_job.split() if len(w) > 3}
    resume_words = set(cleaned_resume.split())
    keyword_score = (len(job_words & resume_words) /
                     len(job_words) * 100) if job_words else 0

    return min(round(0.7 * tf_score + 0.3 * keyword_score, 2), 100.0)


# Skill gap classification

def get_skill_gap(resume_text, job_desc_text):
    # Common tech skills to check
    skill_list = [
        "python", "sql", "machine learning", "deep learning", "nlp",
        "tensorflow", "pytorch", "flask", "django", "fastapi",
        "docker", "git", "aws", "azure", "excel", "power bi",
        "pandas", "numpy", "scikit-learn", "communication",
        "data analysis", "tableau", "java", "javascript", "html", "css",
        "spark", "hadoop", "mongodb", "postgresql", "api", "rest"
    ]

    resume_lower = resume_text.lower()
    job_lower = job_desc_text.lower()

    matched = []
    missing = []
    extra = []

    for skill in skill_list:
        in_resume = skill in resume_lower
        in_job = skill in job_lower

        if in_job and in_resume:
            matched.append(skill)        # skill required AND present
        elif in_job and not in_resume:
            # skill required but MISSING from resume
            missing.append(skill)
        elif in_resume and not in_job:
            # skill in resume but NOT required by job
            extra.append(skill)

    return matched, missing, extra


# Ranking system for multiple candidates

def rank_candidates(candidates_text, job_desc_text):
    cleaned_job = clean_text(job_desc_text)
    results = []  # empty file

    for name, text in candidates_text.items():
        cleaned = clean_text(text)
        tfidf = TfidfVectorizer(ngram_range=(
            1, 2), max_features=5000, stop_words='english')
        # resume text, job desc)
        vecs = tfidf.fit_transform([cleaned, cleaned_job])
        # was re-fitting TF-IDF per candidate
        score = get_hybrid_score(text, job_desc_text)
    results.append({"Candidate": name, "Match Score (%)": score})
    # Sort by score descending — highest match first
    results = sorted(results, key=lambda x: x["Match Score (%)"], reverse=True)
    return results


#  STREAMLIT UI


st.set_page_config(page_title="AI Resume Intelligence System", layout="wide")
st.title("AI Resume Intelligence System 💼")
st.write("Upload resumes and compare with a job description to find the best match.")

# Two tabs: Single Resume vs Multiple Candidates
tab1, tab2 = st.tabs(["Single Resume Analyzer", "Multi-Candidate Ranking"])


# TAB 1 -SINGLE RESUME ANALYSIS

with tab1:
    st.subheader("Upload Resume & Job Description")

    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    job_desc = st.text_area("Enter Job Description", height=200)

    if uploaded_file and job_desc:

        # BULLET 2 - Extract + clean (preprocessing)
        resume_text = extract_text_from_pdf(uploaded_file)
        cleaned_resume = clean_text(resume_text)
        cleaned_job = clean_text(job_desc)

        # BULLET 6 - Vectorize with tuned TF-IDF
        vectors, tfidf = vectorize(cleaned_resume, cleaned_job)

        # Match score
        score = get_hybrid_score(resume_text, job_desc)

        # RESULT DISPLAY

        st.markdown("---")
        st.subheader("Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Match Score", value=f"{score}%")

            if score >= 70:
                st.success("Strong Match — Great fit for this role!")
            elif score >= 40:
                st.warning("Moderate Match — Some gaps exist.")
            else:
                st.error("Weak Match — Resume needs significant improvement.")

        with col2:
            # BULLET 5 - Matplotlib chart: match score gauge bar
            fig, ax = plt.subplots(figsize=(4, 1.5))
            ax.barh(["Match"], [score], color="#4CAF50" if score >=
                    70 else "#FF9800" if score >= 40 else "#F44336")
            ax.barh(["Match"], [100 - score], left=[score], color="#e0e0e0")
            ax.set_xlim(0, 100)
            ax.set_xlabel("Score (%)")
            ax.set_title("Resume Match Score")
            ax.spines[['top', 'right', 'left']].set_visible(False)
            st.pyplot(fig)
            plt.close()

        # Skill Gap Analysis

        st.markdown("---")
        st.subheader("Skill Gap Analysis")

        matched, missing, extra = get_skill_gap(resume_text, job_desc)

        col3, col4, col5 = st.columns(3)

        with col3:
            st.markdown("**Matched Skills**")
            if matched:
                for skill in matched:
                    st.markdown(f"✅ {skill}")
            else:
                st.write("No common skills found.")

        with col4:
            st.markdown("**Missing Skills** *(required by job)*")
            if missing:
                for skill in missing:
                    st.markdown(f"❌ {skill}")
            else:
                st.write("No missing skills — great!")

        with col5:
            st.markdown("**Extra Skills** *(in resume, not required)*")
            if extra:
                for skill in extra:
                    st.markdown(f"➕ {skill}")
            else:
                st.write("None")

        #  - Seaborn chart: skill gap visualization

        st.markdown("---")
        st.subheader("Skill Gap Visualization")

        skill_counts = {
            "Matched": len(matched),
            "Missing": len(missing),
            "Extra": len(extra)
        }

        fig2, ax2 = plt.subplots(figsize=(6, 3))
        colors = ["#4CAF50", "#F44336", "#2196F3"]
        sns.barplot(
            x=list(skill_counts.keys()),
            y=list(skill_counts.values()),
            palette=colors,
            ax=ax2
        )
        ax2.set_title("Skill Gap Summary")
        ax2.set_ylabel("Number of Skills")
        ax2.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig2)
        plt.close()


# TAB 2 - MULTI-CANDIDATE RANKING SYSTEM

with tab2:
    st.subheader("Rank Multiple Candidates")
    st.write("Upload multiple resumes and the system will rank them by job relevance.")

    job_desc_rank = st.text_area(
        "Enter Job Description for Ranking", height=150)
    uploaded_files = st.file_uploader(
        "Upload Multiple Resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files and job_desc_rank:

        candidates = {}
        for f in uploaded_files:
            text = extract_text_from_pdf(f)
            candidates[f.name] = text

        rankings = rank_candidates(candidates, job_desc_rank)

        st.markdown("---")
        st.subheader("Candidate Rankings")

        # Show ranking table
        import pandas as pd
        df = pd.DataFrame(rankings)
        df.index += 1  # rank starts from 1
        df.index.name = "Rank"
        st.dataframe(df, use_container_width=True)

        # BULLET 5 - Seaborn bar chart for candidate ranking
        fig3, ax3 = plt.subplots(figsize=(8, max(3, len(rankings) * 0.8)))
        colors_rank = ["#4CAF50" if r["Match Score (%)"] >= 70
                       else "#FF9800" if r["Match Score (%)"] >= 40
                       else "#F44336" for r in rankings]

        sns.barplot(
            x=[r["Match Score (%)"] for r in rankings],
            y=[r["Candidate"] for r in rankings],
            palette=colors_rank,
            ax=ax3,
            orient='h'
        )
        ax3.set_title("Candidate Match Scores (Ranked)")
        ax3.set_xlabel("Match Score (%)")
        ax3.set_xlim(0, 100)
        ax3.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig3)
        plt.close()

        # Best candidate callout
        best = rankings[0]
        st.success(
            f"Top Candidate: {best['Candidate']} with {best['Match Score (%)']}% match")
