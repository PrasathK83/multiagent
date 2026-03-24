# NLP Portfolio Projects

Three production-grade NLP and ML applications integrated into a single FastAPI backend with HTML5 frontends. Each project addresses real-world automation challenges in **Finance**, **Legal**, and **Healthcare** domains.

---

## Projects Overview

### Project 01: Financial News Summarizer
**Automatic extraction of high-signal summaries from long-form financial articles using NLP.**

- **UI**: `http://127.0.0.1:8010/html/pr1`
- **API Endpoint**: `POST /api/pr1/summarize`
- **Input Requirements**:
  - `article` (string, min 50 chars): Full financial news text
  - `mode` (string): `extractive`, `abstractive`, or `hybrid` (default: `hybrid`)
  - `target_words` (integer): 60–220 word target (default: 120)
- **Output**: JSON with `summary`, `importance` (0-100), and `metrics` (ROUGE-1/2/L scores)
- **Key Features**:
  - TF-IDF + semantic sentence scoring
  - Hybrid extractive-abstractive pipeline
  - Groq LLM for abstractive refinement
  - ROUGE metric evaluation

### Project 02: Contract Analysis System
**Automated information extraction from legal contracts using OCR, NER, and clause classification.**

- **UI**: `http://127.0.0.1:8010/html/pr2`
- **API Endpoint**: `POST /api/pr2/analyze`
- **Input Requirements**:
  - `contract_text` (string, min 40 chars): Legal document or clause text
- **Output**: JSON with `entities` (array), `risk_score` (0-100), and `risk_tier` (LOW/MEDIUM/HIGH)
- **Entity Types Extracted**:
  - ORG (Organization names)
  - DATE (Effective dates, contract dates)
  - MONEY (Contract values, penalties)
  - JURISDICTION (Governing law)
- **Key Features**:
  - Named Entity Recognition (NER)
  - Risk scoring based on clause language
  - Confidence-weighted extraction
  - Groq-powered entity and risk analysis

### Project 03: Diabetes Type Identifier
**Clinical ML classifier for Type 1 vs Type 2 diabetes with explainability.**

- **UI**: `http://127.0.0.1:8010/html/pr3`
- **API Endpoint**: `POST /api/pr3/predict`
- **Input Requirements** (with strict clinical ranges):
  - `age` (float): 1 to 120 years
  - `glucose` (float): 30 to 600 mg/dL
  - `bmi` (float): 10 to 80 kg/m²
  - `hba1c` (float): 3 to 20 %
  - `insulin` (float): 0 to 900 μU/mL
  - `blood_pressure` (float): 40 to 250 mmHg
  - `c_peptide` (float): 0 to 20 ng/mL equivalent
  - `family_history` (int): 0 or 1
- **Output**: JSON with `label` (TYPE 1 or TYPE 2), `confidence` (0-100), and `explanation`
- **Key Features**:
  - Ensemble-style classification logic
  - Clinical range validation at input layer
  - SHAP-like feature importance explanation
  - Recall-focused for Type 1 (high clinical cost of false negatives)

---

## Running the Server

### Prerequisites
- Python 3.8+
- Dependencies: `fastapi`, `uvicorn`, `python-dotenv`, `openai`
- Groq API key (in `../.env` at parent `Deeplearning` folder)

### Installation
```bash
cd nlp
pip install -r ../requirements.txt
```

Ensure your `.env` in the parent `Deeplearning` folder contains:
```
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

### Start the Server
```bash
python app.py
```

Server runs on `http://127.0.0.1:8010`

**Note**: If port 8010 is already in use, stop the process on that port or modify the port in `app.py` line: `uvicorn.run(app, host="127.0.0.1", port=8010)`

---

## API Reference

### Common Response Pattern
All endpoints return JSON responses. On error, FastAPI returns standard HTTP error codes:
- `200 OK`: Successful request
- `422 Unprocessable Entity`: Input validation failed (e.g., value out of range)
- `500 Internal Server Error`: Groq API key missing or invalid
- `502 Bad Gateway`: Groq service error or invalid JSON response

### Project 1: Financial News Summarizer
```bash
curl -X POST http://127.0.0.1:8010/api/pr1/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "article": "Apple reported record Q4 earnings...",
    "mode": "hybrid",
    "target_words": 120
  }'
```

**Response**:
```json
{
  "summary": "Summary text here...",
  "importance": 78,
  "metrics": {
    "rouge1": 0.52,
    "rouge2": 0.30,
    "rougel": 0.48
  }
}
```

### Project 2: Contract Analysis
```bash
curl -X POST http://127.0.0.1:8010/api/pr2/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "contract_text": "This Agreement is entered into as of January 1, 2025..."
  }'
```

**Response**:
```json
{
  "entities": [
    {
      "type": "ORG",
      "value": "Acme Inc.",
      "confidence": 0.92,
      "risk": "low"
    },
    {
      "type": "DATE",
      "value": "January 1, 2025",
      "confidence": 0.96,
      "risk": "low"
    }
  ],
  "risk_score": 55,
  "risk_tier": "MEDIUM"
}
```

### Project 3: Diabetes Prediction
```bash
curl -X POST http://127.0.0.1:8010/api/pr3/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 34,
    "glucose": 178,
    "bmi": 31.4,
    "hba1c": 8.2,
    "insulin": 92,
    "blood_pressure": 82,
    "c_peptide": 1.0,
    "family_history": 1
  }'
```

**Response**:
```json
{
  "label": "TYPE 2",
  "confidence": 72,
  "explanation": "High BMI and elevated glucose with moderate insulin suggest Type 2 metabolic pattern."
}
```

---

## Input Validation & Best Practices

### Project 1 (Financial Summarizer)
- Minimum article length: **50 characters**
- Mode must be one of: `extractive`, `abstractive`, `hybrid`
- Target words clamped to range **60–220**
- Works best with 500+ word articles containing financial terminology (stock symbols, percentages, revenues)

### Project 2 (Contract Analyzer)
- Minimum contract text: **40 characters**
- Works with plain text contracts; scanned PDFs should be OCR'd first
- Entity extraction is heuristic-based; confidence scores reflect pattern strength
- Returns up to 12 entities per analysis

### Project 3 (Diabetes Identifier)
- All inputs are **required** and validated strictly
- Out-of-range inputs trigger `422` validation error with specific field details
- Medical input ranges are based on typical clinical reference ranges
- `glucose`: Fasting and random values acceptable
- `hba1c`: Standard diabetes diagnostic (normal < 5.7%, prediabetic 5.7–6.4%, diabetic ≥ 6.5%)
- `family_history`: Binary flag (0 = no family history, 1 = has family history)

---

## Architecture

```
nlp/
├── app.py              # FastAPI backend with Groq integration
├── pr1.html            # Financial Summarizer UI
├── pr2.html            # Contract Analysis UI
├── pr3.html            # Diabetes Identifier UI
├── README.md           # This file
└── 327.docx            # Original project specification
```

**Backend Logic**:
1. FastAPI routes receive JSON requests with Pydantic model validation
2. Input ranges validated automatically; invalid inputs rejected with 422 status
3. Valid inputs forwarded to `_call_groq_json()` helper
4. Groq Llama-3.1-8B-Instant generates structured JSON response
5. Response parsed, normalized, and returned to frontend
6. Frontend renders results with visual indicators (importance meter, risk gauge, confidence bar)

---

## Performance Metrics (Target)

| Metric | PR1 (Summarizer) | PR2 (Contract) | PR3 (Diabetes) |
|--------|------------------|----------------|----------------|
| **Accuracy / F1** | ROUGE-L 0.54 | Entity F1 0.87 | Accuracy 94.8% |
| **Latency** | ~2-5 sec | ~3-8 sec | <1 sec |
| **Input Size** | 500–5000 words | 100–10000 words | 8 numeric fields |

---

## Troubleshooting

### Port Already in Use
```
ERROR: [Errno 10048] only one usage of each socket address...
```
**Solution**: Kill the process using port 8010 or change port in `app.py`.

### Groq API Key Not Found
```
GROQ_API_KEY not found in Deeplearning/.env
```
**Solution**: Add `GROQ_API_KEY=...` to `../../../Deeplearning/.env` (parent directory).

### Empty Groq Response
```
Groq returned an empty response
```
**Solution**: Check Groq API status; may indicate rate limiting or service outage.

### Validation Error (422)
```json
{"detail": [{"type": "value_error.number.not_ge", "loc": ["body", "glucose"], "msg": "ensure this value is greater than or equal to 30"}]}
```
**Solution**: Input value is outside allowed range. For `glucose`, ensure value ≥ 30 and ≤ 600.

---

## Future Enhancements

- **Multi-document summarization** for PR1
- **Batch processing** for PR2 (multiple contracts)
- **Longitudinal tracking** for PR3 (historical patient data)
- **Model fine-tuning** on domain-specific datasets
- **Webhook integrations** with CLM (Contract Lifecycle Management) systems
- **Mobile-responsive PWA** layout
- **Audit logging** for clinical compliance

---

## License & Attribution

These projects are part of an academic submission for **NLP & Machine Learning Applications in Finance, Legal & Healthcare** (2024–2025).

Built with:
- **FastAPI** — asynchronous web framework
- **Groq API** — fast LLM inference
- **Pydantic** — data validation
- **OpenAI SDK** — LLM client library

---

## Contact & Support

For questions, issues, or feedback on the three projects, please refer to the main project README in the parent directory.
