from src.models import AnalysisModels as Models
from typing import Dict


def evaluate(payload: Dict) -> Dict:
    texts = payload["texts"]
    model_name = payload["model_name"]
    models = Models("configs/models/analysis-models.yaml", "models/analysis-models/")
    model = getattr(models, model_name)
    scores, preds = model(texts)

    return {"scores": scores, "predictions": preds}
