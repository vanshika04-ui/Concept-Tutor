# Concept Tutor

LLM-powered tutoring API that diagnoses misconceptions and teaches concepts.

## What it does
- `/start` - Create learning session
- `/check/{session_id}` - Submit wrong answer, get diagnosis
- `/teach` - Get mini-lesson for any concept
- `/progress/{session_id}` - See student's concept history

## Run locally
```bash
pip install -r requirements.txt
set GROQ_API_KEY=your_key_here  # Windows
uvicorn app:app --reload

