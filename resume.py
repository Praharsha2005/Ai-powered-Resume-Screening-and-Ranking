import pandas as pd
import re
import nltk
import spacy
import subprocess
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text
from docx import Document
import streamlit as st
from textstat import flesch_reading_ease

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Ensure the SpaCy model is installed
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(file):
    return extract_text(file)

def extract_text_from_docx(file):
    doc = Document(file)
    return '\n'.join([para.text for para in doc.paragraphs])

def clean_text(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def extract_skills(text):
    skills = set()
    doc = nlp(text)
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            skills.add(token.text.lower())
    return skills

def extract_experience(text):
    exp_match = re.search(r'(\d+)\s*(?:years|yrs|year|yr)\s*(?:experience|exp)?', text, re.IGNORECASE)
    return int(exp_match.group(1)) if exp_match else 0

def calculate_keyword_density(text, keywords):
    words = text.split()
    keyword_count = sum(1 for word in words if word in keywords)
    return keyword_count / len(words) if words else 0

def get_readability_score(text):
    return flesch_reading_ease(text)

st.title("📄 AI-Powered Resume Screening and Ranking System")
uploaded_files = st.file_uploader("📂 Upload Resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
job_description = st.text_area("📝 Enter Job Description")

if st.button("🔍 Process Resumes"):
    if uploaded_files and job_description:
        cleaned_job_desc = clean_text(job_description)
        job_skills = extract_skills(job_description)
        resumes, resume_names, experiences, skill_matches, keyword_densities, readability_scores = [], [], [], [], [], []

        for file in uploaded_files:
            text = extract_text_from_pdf(file) if file.type == "application/pdf" else extract_text_from_docx(file)
            cleaned_text = clean_text(text)
            resumes.append(cleaned_text)
            resume_names.append(file.name)
            experiences.append(extract_experience(text))
            resume_skills = extract_skills(text)
            skill_match_count = len(job_skills.intersection(resume_skills))
            skill_matches.append(skill_match_count)
            keyword_densities.append(calculate_keyword_density(cleaned_text, job_skills))
            readability_scores.append(get_readability_score(text))

        vectorizer = TfidfVectorizer()
        job_desc_vector = vectorizer.fit_transform([cleaned_job_desc])
        resume_vectors = vectorizer.transform(resumes)
        similarity_scores = cosine_similarity(job_desc_vector, resume_vectors).flatten()

        final_scores = [((0.3 * sim) + (0.3 * (exp / 10)) + (0.2 * (skills / len(job_skills) if job_skills else 0)) + (0.2 * density)) 
                        for sim, exp, skills, density in zip(similarity_scores, experiences, skill_matches, keyword_densities)]
        
        ranked_resumes = pd.DataFrame({
            'Resume': resume_names,
            'Experience (years)': experiences,
            'Similarity Score': similarity_scores,
            'Skill Match Count': skill_matches,
            'Keyword Density': keyword_densities,
            'Readability Score': readability_scores,
            'Final Score': final_scores
        }).sort_values(by='Final Score', ascending=False)

        st.write("📊 **Ranked Resumes:**")
        st.dataframe(ranked_resumes)
        csv = ranked_resumes.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Ranked Resumes", csv, "ranked_resumes.csv", "text/csv")
    else:
        st.error("⚠️ Please upload resumes and enter a job description.")
