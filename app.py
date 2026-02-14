import streamlit as st
from utils import extract_text_from_pdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Intelligent Resume Screening",
    layout="wide"
)

# ---------------- SESSION STATE ----------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# ---------------- TOGGLE BUTTON ----------------
col1, col2 = st.columns([8, 1])
with col2:
    if st.button("🌙 / ☀️"):
        st.session_state.dark_mode = not st.session_state.dark_mode

# ---------------- THEME CSS ----------------
if st.session_state.dark_mode:
    st.markdown("""
    <style>
        .stApp {
            background-color: #0f172a;
            color: #e5e7eb;
        }
        h1, h2, h3 {
            color: #f8fafc;
        }
        textarea, .stTextArea textarea {
            background-color: #020617 !important;
            color: #f8fafc !important;
        }
        .stButton>button {
            background-color: #2563eb;
            color: white;
        }
        .dataframe {
            background-color: #020617;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        .stApp {
            background-color: #ffffff;
            color: #000000;
        }
        h1, h2, h3 {
            color: #000000;
        }
    </style>
    """, unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown(
    "<h1 style='text-align: center;'>📄 Intelligent Resume Screening System</h1>",
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- JOB DESCRIPTION ----------------
st.subheader("📌 Job Description")

job_desc = st.text_area(
    "Paste Job Description Here",
    height=220
)

# ---------------- UPLOAD RESUMES ----------------
st.subheader("📂 Upload Resumes (PDF)")

uploaded_files = st.file_uploader(
    "Upload multiple resumes",
    type=["pdf"],
    accept_multiple_files=True
)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- BUTTON ----------------
if st.button("🔍 Screen Resumes", use_container_width=True):

    if not job_desc or not uploaded_files:
        st.warning("⚠️ Please provide job description and upload resumes.")
    else:
        resumes = []
        names = []

        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            resumes.append(text)
            names.append(file.name)

        documents = [job_desc] + resumes

        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(documents)

        scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        results = pd.DataFrame({
            "Resume": names,
            "Match Score (%)": (scores * 100).round(2)
        }).sort_values(by="Match Score (%)", ascending=False)

        st.markdown("---")
        st.subheader("📊 Screening Results")

        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.dataframe(
                results,
                use_container_width=True,
                height=300
            )
