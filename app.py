from typing import Dict, List
import uuid
from fastapi import FastAPI
from pydantic import BaseModel
import os
from groq import Groq

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