from typing import Dict, List
import uuid
from fastapi import FastAPI,UploadFile, File
from pydantic import BaseModel
import os
from groq import Groq
from rag import doc_store
import pypdf
import io

# SECURE: Load from environment
client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))

app = FastAPI()
sessions: Dict[str, List[str]] = {}

class Answer(BaseModel):
    problem: str
    wrong_answer: str

@app.post("/start")
def start_session():
    session_id = str(uuid.uuid4())[:8]
    sessions[session_id] = []
    return {"session_id": session_id, "message": "Session started"}

@app.post("/check/{session_id}")
def check_answer(session_id: str, ans: Answer):
    history = sessions.get(session_id, [])
    history_text = ", ".join(history) if history else "none"
    
    prompt = f"""Analyze the error. Reply in EXACTLY this format:
CONCEPT: [2-3 word concept name]
WHY: [one sentence explanation]

Student history: {history_text}
Problem: {ans.problem}
Wrong answer: {ans.wrong_answer}"""
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.1
    )
    
    text = response.choices[0].message.content
    concept = "unknown"
    explanation = text
    
    if "CONCEPT:" in text:
        lines = text.split("\n")
        for line in lines:
            if line.startswith("CONCEPT:"):
                concept = line.replace("CONCEPT:", "").strip().lower().replace(" ", "_")
            if line.startswith("WHY:"):
                explanation = line.replace("WHY:", "").strip()
    
    sessions[session_id].append(concept)
    
    return {
        "concept": concept,
        "explanation": explanation,
        "concepts_history": sessions[session_id]
    }

@app.post("/teach")
def teach_concept(concept: str):
    prompt = f"""Teach '{concept}' in this format:
INTUITION: One sentence.
EXAMPLE: One concrete example.
COMMON_MISTAKE: One frequent error.

No extra text."""
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.1
    )
    
    text = response.choices[0].message.content
    result = {"concept": concept, "raw": text}
    
    for line in text.split("\n"):
        if line.startswith("INTUITION:"):
            result["intuition"] = line.replace("INTUITION:", "").strip()
        elif line.startswith("EXAMPLE:"):
            result["example"] = line.replace("EXAMPLE:", "").strip()
        elif line.startswith("COMMON_MISTAKE:"):
            result["common_mistake"] = line.replace("COMMON_MISTAKE:", "").strip()
    
    return result

@app.get("/progress/{session_id}")
def get_progress(session_id: str):
    history = sessions.get(session_id, [])
    return {
        "session_id": session_id,
        "total_mistakes": len(history),
        "concepts": list(set(history))
    }


@app.post("/upload-pdf/{session_id}")
async def upload_pdf(session_id: str, file: UploadFile = File(...)):
    """
    Upload PDF and create searchable knowledge base for this session
    """
    # Read PDF
    contents = await file.read()
    pdf = pypdf.PdfReader(io.BytesIO(contents))
    
    # Chunk by page (simple approach)
    chunks = []
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text.strip():  # Skip empty pages
            chunks.append({
                "id": f"{session_id}_page_{i}",
                "text": text[:1000],  # First 1000 chars per page
                "page": i + 1,
                "source": file.filename
            })
    
    # Store in vector DB
    count = doc_store.add_document(session_id, chunks)
    
    return {
        "session_id": session_id,
        "filename": file.filename,
        "pages_indexed": count,
        "status": "ready for questions"
    }

@app.post("/ask-pdf/{session_id}")
async def ask_pdf(session_id: str, question: str):
    """
    Ask question based on uploaded PDF
    """
    try:
        # Retrieve relevant context
        results = doc_store.query(session_id, question, n_results=2)
        
        # Build context
        context = "\n\n---\n\n".join(results["documents"])
        sources = [f"Page {m['page']}" for m in results["metadatas"]]
        
        # Ask LLM with context
        prompt = f"""Answer the question using ONLY the provided context.
If the answer is not in the context, say "I don't know based on this document."

Context:
{context}

Question: {question}

Answer:"""
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.1
        )
        
        answer = response.choices[0].message.content
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "confidence": "high" if "I don't know" not in answer else "low"
        }
        
    except Exception as e:
        return {
            "error": "No PDF uploaded for this session or query failed",
            "detail": str(e)
        }
    
    @app.post("/evaluate-rag/{session_id}")
async def evaluate_rag(session_id: str, question: str, expected_answer: str):
    """Simple RAG evaluation"""
    result = await ask_pdf(session_id, question)
    
    # Check if answer contains key terms from expected
    expected_words = set(expected_answer.lower().split())
    answer_words = set(result["answer"].lower().split())
    overlap = len(expected_words & answer_words) / len(expected_words)
    
    return {
        "question": question,
        "retrieved_sources": result["sources"],
        "answer": result["answer"],
        "expected": expected_answer,
        "term_overlap": f"{overlap:.0%}",
        "retrieval_worked": overlap > 0.5
    }