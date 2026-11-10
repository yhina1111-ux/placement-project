from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib, pandas as pd

app = FastAPI(title="Placement Predictor")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

pipe = joblib.load("placement_model.pkl")
columns = joblib.load("model_columns.pkl")
scaler, lr = pipe[0], pipe[-1]


class Student(BaseModel):
    cgpa: float = Field(ge=0, le=10)
    internships: int = Field(ge=0)
    projects: int = Field(ge=0)
    workshops: int = Field(ge=0)
    aptitude: float = Field(ge=0, le=100)
    soft_skills: float = Field(ge=0, le=5)
    extracurricular: bool
    training: bool
    ssc: float = Field(ge=0, le=100)
    hsc: float = Field(ge=0, le=100)


def to_df(s: Student):
    row = {
        "CGPA": s.cgpa,
        "Internships": s.internships,
        "Projects": s.projects,
        "Workshops/Certifications": s.workshops,
        "AptitudeTestScore": s.aptitude,
        "SoftSkillsRating": s.soft_skills,
        "ExtracurricularActivities": int(s.extracurricular),
        "PlacementTraining": int(s.training),
        "SSC_Marks": s.ssc,
        "HSC_Marks": s.hsc,
    }
    return pd.DataFrame([row])[columns]


@app.get("/")
def home():
    return {"message": "Placement predictor is running"}


@app.post("/predict")
def predict(s: Student):
    df = to_df(s)
    prob = float(pipe.predict_proba(df)[0][1])
    contrib = lr.coef_[0] * scaler.transform(df)[0]
    explanation = sorted(
        [{"feature": f, "impact": round(float(c), 3)}
         for f, c in zip(columns, contrib)],
        key=lambda x: abs(x["impact"]), reverse=True)
    return {
        "placed": prob >= 0.5,
        "probability": round(prob * 100, 1),
        "explanation": explanation,
    }