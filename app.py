# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime
import uvicorn
import os
from model_loader import load_models, get_scores

origins = [
    "http://127.0.0.1",
    "http://localhost",
    os.environ.get("CORS_ORIGINS", "*") # Get the live URL from Render
]



app = FastAPI(title="Lead Scoring API")
leads_db = []



@app.on_event("startup")
def startup_event():
    if not load_models(): print("WARNING: Model loading failed.")

class Lead(BaseModel):
    email: EmailStr; page_views: int = Field(..., ge=0); form_submissions: int = Field(..., ge=0)
    previous_interactions: int = Field(..., ge=0); time_spent_website_secs: int = Field(..., ge=0)
    credit_score: int = Field(..., ge=300, le=850); income_inr: int = Field(..., ge=0)
    profession: str; city: str; property_type_preference: str; lead_source: str
    gender: str; age_group: str; education: str; family_background: str; comments: str = ""

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/score")
def score_lead(lead: Lead):
    try:
        initial_score, final_score = get_scores(lead.dict())
        result = { 
            "email": lead.email, "comments": lead.comments, 
            "initial_score": round(initial_score, 2), 
            "reranked_score": round(final_score, 2), # This will now have the correct value
            "timestamp": datetime.now().isoformat() 
        }
        leads_db.append(result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model execution error: {str(e)}")

@app.get("/leads")
def get_leads():
    return sorted(leads_db, key=lambda x: x['timestamp'], reverse=True)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)