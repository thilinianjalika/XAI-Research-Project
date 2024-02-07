from typing import Dict
from handlers import evaluate, analyze
import traceback


def handler(event: Dict, context: Dict):
    task = event["task"]
    payload = event["payload"]
    try:
        if task == "evaluation":
            body = evaluate(payload)
            return {"status": 200, "body": body}
        elif task == "analysis":
            body = analyze(payload)
            return {"status": 200, "body": body}
        else:
            body = "Invocation error"
            return {"status": 400, "body": body}
    except Exception as e:
        body = str(e)
        print("Error:", body)
        print("Input:", event)
        traceback.print_exc()
        return {"status": 500, "body": body}
