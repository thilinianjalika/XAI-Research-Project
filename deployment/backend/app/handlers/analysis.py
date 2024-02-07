from typing import Dict
from src import TestBench

tb_kwargs = {
    "svm": {"cf_generator_config": "configs/models/wf-cf-generator.yaml"},
    "knn": {"cf_generator_config": "configs/models/wf-cf-generator.yaml"},
    "rf": {
        "threshold_classifier": 0.49339999999983775,
        "max_iter": 50,
        "time_maximum": 120,
    },
    "lr": {
        "threshold_classifier": 0.49179999999978463,
        "max_iter": 50,
        "time_maximum": 120,
    },
}


def analyze(payload: Dict):
    model_name = payload["model_name"]
    configurations = payload["configurations"]
    prompt = payload["prompt"]
    variations = payload["variations"]
    tb = TestBench(
        model_path=f"models/analysis-models/{model_name}.pkl",
        vectorizer_path="models/analysis-models/tfidf.pkl",
        analyzer_name=model_name,
        **tb_kwargs[model_name],
    )
    reports = tb(configurations, prompt, variations)
    reports = "\n\n".join(reports)
    return reports
