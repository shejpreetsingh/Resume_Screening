from flask import Flask, request, render_template, jsonify
import os
import pdfplumber

import docx
import pytesseract
from PIL import Image
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

app = Flask(__name__)

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''.join(page.extract_text() for page in pdf.pages)
    return text

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    text = '\n'.join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_image(image_path):
    text = pytesseract.image_to_string(Image.open(image_path))
    return text

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

def match_resume_to_job(resume_texts, job_description):
    texts = resume_texts + [job_description]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    return similarity_scores.flatten()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/match', methods=['POST'])
def match():

    job_description = request.form['job_description']
    job_description_cleaned = preprocess_text(job_description)

    uploaded_files = request.files.getlist('resumes')
    resumes = []

    for file in uploaded_files:
        filename = file.filename
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)

        elif filename.endswith('.docx'):
            text = extract_text_from_docx(file_path)

        elif filename.endswith(('.png', '.jpg', '.jpeg')):
            text = extract_text_from_image(file_path)
        else:
            continue

        cleaned_text = preprocess_text(text)
        resumes.append(cleaned_text)

    scores = match_resume_to_job(resumes, job_description_cleaned)

    results = [{"resume": uploaded_files[i].filename, "score": f"{scores[i] * 100:.2f}"} for i in range(len(scores))]

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)