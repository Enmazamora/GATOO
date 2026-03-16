import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from pypdf import PdfReader
from bs4 import BeautifulSoup
from groq import Groq
# --- Environment Fixes (Vercel) ---
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

app = Flask(__name__)
CORS(app)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
PDF_PATH = os.path.join(PROJECT_ROOT, "gato.pdf")
HTML_PATH = os.path.join(PROJECT_ROOT, "index.html")
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 3

# --- Groq Configuration ---
# The user will need to provide this key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")

# --- Text Extraction ---
def get_pdf_text(path):
    if not os.path.exists(path):
        print(f"ERROR: PDF not found at {path}")
        return ""
    try:
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def get_html_text(path):
    if not os.path.exists(path):
        print(f"ERROR: HTML not found at {path}")
        return ""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            main = soup.find('main')
            footer = soup.find('footer')
            return ((main.get_text(separator=' ') if main else "") + "\n" +
                    (footer.get_text(separator=' ') if footer else ""))
    except Exception as e:
        print(f"Error reading HTML: {e}")
        return ""

def chunk_text(text, size, overlap):
    text = re.sub(r'\s+', ' ', text).strip()
    chunks = []
    if len(text) <= size:
        return [text] if text else []
    for i in range(0, len(text), size - overlap):
        chunk = text[i:i + size]
        if len(chunk) > 60:
            chunks.append(chunk)
    return chunks

# --- Lightweight TF-IDF Implementation ---
import math
from collections import Counter

def get_tokens(text):
    return re.findall(r'\w+', text.lower())

def compute_tfidf(chunks):
    total_docs = len(chunks)
    doc_tokens = [get_tokens(c) for c in chunks]
    
    # DF (Document Frequency)
    df = Counter()
    for tokens in doc_tokens:
        unique_tokens = set(tokens)
        for t in unique_tokens:
            df[t] += 1
            
    # IDF
    idf = {t: math.log(total_docs / (count + 1)) for t, count in df.items()}
    
    # TF-IDF vectors for docs
    doc_vectors = []
    for tokens in doc_tokens:
        tf = Counter(tokens)
        vec = {t: (count / len(tokens)) * idf.get(t, 0) for t, count in tf.items()}
        doc_vectors.append(vec)
        
    return doc_vectors, idf

def cosine_similarity_lite(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    if not denominator:
        return 0.0
    return float(numerator) / denominator

# --- Initialize knowledge base ---
print("Consolidando el conocimiento del Oráculo (via Groq)...")
all_chunks = []

pdf_text = get_pdf_text(PDF_PATH)
if pdf_text:
    pdf_chunks = chunk_text(pdf_text, CHUNK_SIZE, CHUNK_OVERLAP)
    all_chunks.extend(pdf_chunks)
    print(f"PDF: {len(pdf_chunks)} fragmentos cargados.")

html_text = get_html_text(HTML_PATH)
if html_text:
    html_chunks = chunk_text(html_text, CHUNK_SIZE, CHUNK_OVERLAP)
    all_chunks.extend(html_chunks)
    print(f"HTML: {len(html_chunks)} fragmentos cargados.")

# Build custom vectors
if all_chunks:
    DOC_VECTORS, IDF = compute_tfidf(all_chunks)
    print(f"Total: {len(all_chunks)} fragmentos listos para el Oráculo.")
else:
    print("ALERTA: Base de conocimiento vacía.")
    DOC_VECTORS, IDF = [], {}

def retrieve_context(query, top_k=TOP_K):
    if not all_chunks:
        return []
    
    tokens = get_tokens(query)
    tf = Counter(tokens)
    query_vec = {t: (count / len(tokens)) * IDF.get(t, 0) for t, count in tf.items()}
    
    similarities = [cosine_similarity_lite(query_vec, doc_vec) for doc_vec in DOC_VECTORS]
    
    results = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:top_k]
    return [all_chunks[i] for i, sim in results if sim > 0.01]

# --- Groq AI ---
def ask_groq(query, context_chunks):
    if not GROQ_API_KEY:
        return "No se ha configurado la API Key de Groq. Por favor proporcione una para continuar."
    try:
        client = Groq(api_key=GROQ_API_KEY)
        context = "\n\n---\n\n".join(context_chunks)
        
        prompt = f"""Eres el Oráculo de Plutón, un espíritu oscuro y sabio que habita en la historia de "El Gato Negro" de Edgar Allan Poe. 
Respondes en español, con un tono evocador y literario, basándote únicamente en el contexto proporcionado.
Si la respuesta no se puede deducir del contexto, dilo honestamente con tus palabras de espectro.

CONTEXTO DEL CUENTO:
{context}

PREGUNTA DEL MORTAL:
{query}"""

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Eres un asistente experto en literatura gótica, respondiendo como un oráculo espectral."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=800,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error en Groq: {e}")
        return f"El Oráculo sufre un oscuro error con Groq: {str(e)}"

# --- Flask endpoint ---
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json or {}
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({"answer": "Habla, mortal. ¿Qué deseas saber?"})

    context_chunks = retrieve_context(query)

    if not context_chunks:
        answer = "Las sombras no traen respuesta sobre eso. Pregunta sobre el cuento o sobre su oscuro autor."
    else:
        answer = ask_groq(query, context_chunks)

    return jsonify({"answer": answer})

if __name__ == '__main__':
    print("\n[AVISO] El Oráculo está listo para usar Groq.")
    print("   Asegúrate de configurar GROQ_API_KEY.\n")
    app.run(port=5000, debug=False)
