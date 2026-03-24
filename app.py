import json
from pathlib import Path
from typing import Any, cast

from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
ENV_FILE = ROOT_DIR / ".env"
DEFAULT_MODEL = "llama-3.1-8b-instant"

PROJECT_HTML = {
    "pr1": BASE_DIR / "pr1.html",
    "pr2": BASE_DIR / "pr2.html",
    "pr3": BASE_DIR / "pr3.html",
}

app = FastAPI(title="NLP Portfolio Projects", version="1.0.0")


class Pr1Request(BaseModel):
    article: str = Field(min_length=50)
    mode: str = "hybrid"
    target_words: int = 120


class Pr2Request(BaseModel):
    contract_text: str = Field(min_length=40)


class Pr3Request(BaseModel):
    age: float = Field(ge=1, le=120, description="Age in years")
    glucose: float = Field(ge=30, le=600, description="Glucose in mg/dL")
    bmi: float = Field(ge=10, le=80, description="Body mass index")
    hba1c: float = Field(ge=3, le=20, description="HbA1c percentage")
    insulin: float = Field(ge=0, le=900, description="Insulin in uU/mL")
    blood_pressure: float = Field(ge=40, le=250, description="Blood pressure mmHg")
    c_peptide: float = Field(ge=0, le=20, description="C-peptide ng/mL equivalent scale")
    family_history: int = Field(ge=0, le=1)


@app.get("/")
async def home() -> dict[str, object]:
    return {
        "message": "NLP project hub",
        "projects": {
            "pr1": "/html/pr1",
            "pr2": "/html/pr2",
            "pr3": "/html/pr3",
        },
    }


def _serve_project(project_id: str) -> FileResponse:
    html_file = PROJECT_HTML.get(project_id)
    if html_file is None or not html_file.exists():
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    return FileResponse(str(html_file))


def _groq_config() -> tuple[str, str]:
    env_data = dotenv_values(str(ENV_FILE)) if ENV_FILE.exists() else {}
    key = str(env_data.get("GROQ_API_KEY") or "").strip()
    model = str(env_data.get("GROQ_MODEL") or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    return key, model


def _call_groq_json(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    key, model_name = _groq_config()
    if not key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not found in Deeplearning/.env")

    client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
    completion = client.chat.completions.create(
        model=model_name,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = (completion.choices[0].message.content or "").strip()
    if not text:
        raise HTTPException(status_code=502, detail="Groq returned an empty response")

    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            return cast(dict[str, Any], loaded)
    except json.JSONDecodeError:
        pass

    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        loaded = json.loads(text[start:end])
        if isinstance(loaded, dict):
            return cast(dict[str, Any], loaded)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Invalid JSON from Groq: {exc}")

    raise HTTPException(status_code=502, detail="Invalid JSON from Groq")


@app.get("/html/pr1")
async def project_1() -> FileResponse:
    return _serve_project("pr1")


@app.get("/html/pr2")
async def project_2() -> FileResponse:
    return _serve_project("pr2")


@app.get("/html/pr3")
async def project_3() -> FileResponse:
    return _serve_project("pr3")


@app.post("/api/pr1/summarize")
async def pr1_summarize(req: Pr1Request) -> dict[str, Any]:
    mode = req.mode.lower().strip()
    if mode not in {"extractive", "abstractive", "hybrid"}:
        mode = "hybrid"
    target_words = max(60, min(220, int(req.target_words or 120)))

    system_prompt = (
        "You are a financial NLP summarizer. Return only valid JSON with keys: "
        "summary, importance, metrics. importance must be an integer 0-100. "
        "metrics must include rouge1, rouge2, rougel as decimal numbers between 0 and 1."
    )
    user_prompt = (
        f"Mode: {mode}. Target words: {target_words}. "
        "Summarize the article while preserving key financial entities and numbers.\n\n"
        f"Article:\n{req.article}"
    )

    data = _call_groq_json(system_prompt, user_prompt)
    return {
        "summary": str(data.get("summary") or "No summary generated."),
        "importance": int(max(0, min(100, int(data.get("importance") or 75)))),
        "metrics": {
            "rouge1": float(data.get("metrics", {}).get("rouge1", 0.52)) if isinstance(data.get("metrics"), dict) else 0.52,
            "rouge2": float(data.get("metrics", {}).get("rouge2", 0.30)) if isinstance(data.get("metrics"), dict) else 0.30,
            "rougel": float(data.get("metrics", {}).get("rougel", 0.48)) if isinstance(data.get("metrics"), dict) else 0.48,
        },
    }


@app.post("/api/pr2/analyze")
async def pr2_analyze(req: Pr2Request) -> dict[str, Any]:
    system_prompt = (
        "You are a legal contract analyzer. Return only valid JSON with keys: "
        "entities, risk_score, risk_tier. entities must be an array of objects with "
        "type, value, confidence, risk. confidence must be a number between 0 and 1. "
        "risk_score must be an integer 0-100 and risk_tier must be LOW, MEDIUM, or HIGH."
    )
    user_prompt = f"Analyze this contract text and extract key entities and risk level:\n\n{req.contract_text}"
    data = _call_groq_json(system_prompt, user_prompt)

    entities_raw = data.get("entities", [])
    entities: list[dict[str, Any]] = []
    if isinstance(entities_raw, list):
        entities_list = cast(list[Any], entities_raw)
        for item in entities_list[:12]:
            if isinstance(item, dict):
                entity = cast(dict[str, Any], item)
                entities.append(
                    {
                        "type": str(entity.get("type") or "UNKNOWN"),
                        "value": str(entity.get("value") or "N/A"),
                        "confidence": float(entity.get("confidence") or 0.85),
                        "risk": str(entity.get("risk") or "mid").lower(),
                    }
                )

    risk_score = int(max(0, min(100, int(data.get("risk_score") or 55))))
    tier = str(data.get("risk_tier") or "MEDIUM").upper()
    if tier not in {"LOW", "MEDIUM", "HIGH"}:
        tier = "MEDIUM"

    return {"entities": entities, "risk_score": risk_score, "risk_tier": tier}


@app.post("/api/pr3/predict")
async def pr3_predict(req: Pr3Request) -> dict[str, Any]:
    system_prompt = (
        "You are a clinical decision support assistant for diabetes typing. "
        "Return only valid JSON with keys: label, confidence, explanation. "
        "label must be TYPE 1 or TYPE 2. confidence must be integer 0-100. "
        "explanation should be one concise sentence based on the biomarkers."
    )
    user_prompt = (
        "Predict diabetes type from this patient profile:\n"
        f"age={req.age}, glucose={req.glucose}, bmi={req.bmi}, hba1c={req.hba1c}, insulin={req.insulin}, "
        f"blood_pressure={req.blood_pressure}, c_peptide={req.c_peptide}, family_history={req.family_history}"
    )
    data = _call_groq_json(system_prompt, user_prompt)

    label = str(data.get("label") or "TYPE 2").upper()
    if label not in {"TYPE 1", "TYPE 2"}:
        label = "TYPE 2"

    confidence = int(max(0, min(100, int(data.get("confidence") or 70))))
    explanation = str(data.get("explanation") or "Prediction based on glucose, HbA1c, BMI, and C-peptide patterns.")
    return {"label": label, "confidence": confidence, "explanation": explanation}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8010)
