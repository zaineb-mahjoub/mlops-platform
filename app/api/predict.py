from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import mlflow.pyfunc
import pandas as pd
import tempfile
import sys
from pathlib import Path
from fastapi import UploadFile, File
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
from rag import answer_question, index_document
from extraction import extract_text
from groq_extractor import extract_features as extract_project_spec
from feature_engineering import map_to_features

router = APIRouter()

FEATURE_COLS = [
    "num_functional_requirements", "num_integrations", "requires_web_component",
    "requires_mobile_component", "requires_ai_ml_component", "requires_cloud_infra",
    "requires_data_pipeline", "num_technologies_mentioned", "has_deadline",
    "deadline_days", "complexity_keyword_count", "requirement_text_length"
]

class ProjectFeatures(BaseModel):
    num_functional_requirements: int = Field(ge=0)
    num_integrations: int = Field(ge=0)
    requires_web_component: int = Field(ge=0, le=1)
    requires_mobile_component: int = Field(ge=0, le=1)
    requires_ai_ml_component: int = Field(ge=0, le=1)
    requires_cloud_infra: int = Field(ge=0, le=1)
    requires_data_pipeline: int = Field(ge=0, le=1)
    num_technologies_mentioned: int = Field(ge=0)
    has_deadline: int = Field(ge=0, le=1)
    deadline_days: float = Field(ge=0)
    complexity_keyword_count: int = Field(ge=0)
    requirement_text_length: int = Field(ge=0)

_classifier = None
_regressor = None
_last_document_text = None
_last_document_id = None

def get_classifier():
    global _classifier
    if _classifier is None:
        try:
            _classifier = mlflow.pyfunc.load_model("models:/project_category_classifier@champion")
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Classifier model not available: {e}")
    return _classifier

def get_regressor():
    global _regressor
    if _regressor is None:
        try:
            _regressor = mlflow.pyfunc.load_model("models:/effort_regressor@champion")
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Regressor model not available: {e}")
    return _regressor

def to_frame(features: ProjectFeatures) -> pd.DataFrame:
    return pd.DataFrame([features.model_dump()])[FEATURE_COLS]

@router.post("/classify")
def classify(features: ProjectFeatures):
    model = get_classifier()
    X = to_frame(features)
    pred = model.predict(X)
    return {"project_category": str(pred[0])}

@router.post("/predict-effort")
def predict_effort(features: ProjectFeatures):
    model = get_regressor()
    X = to_frame(features)
    pred = model.predict(X)
    return {"estimated_effort_person_days": round(float(pred[0]), 1)}
@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        raw_text = extract_text(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    global _last_document_text
    _last_document_text = raw_text

    try:
        spec = extract_project_spec(raw_text)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=f"Extraction failed: {e}")

    feature_dict = map_to_features(spec, raw_text)
    features = ProjectFeatures(**feature_dict)
    X = to_frame(features)

    category = get_classifier().predict(X)[0]
    effort = get_regressor().predict(X)[0]

    return {
        "project_specification": spec.model_dump(),
        "classification": {"project_category": str(category)},
        "regression": {"estimated_effort_person_days": round(float(effort), 1)},
    }
class AskRequest(BaseModel):
    question: str

@router.post("/ask")
def ask(request: AskRequest):
    if _last_document_id is None:
        raise HTTPException(
            status_code=400,
            detail="No document has been analyzed yet. Call /analyze first."
        )

    answer = answer_question(
        _last_document_id,
        request.question
    )

    return {"answer": answer}
