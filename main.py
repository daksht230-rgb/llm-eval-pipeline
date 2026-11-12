from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from adversarial_tester import AdvancedLLMEvaluator
from typing import List

app = FastAPI(
    title="Enterprise LLM Evaluation & Adversarial Benchmarking API",
    version="2.0.0",
    description="Production-grade automated evaluation harness for AI model performance and safety metrics."
)

evaluator = AdvancedLLMEvaluator()

class TestCaseSchema(BaseModel):
    prompt: str = Field(..., example="Explain quantum computing in 1 sentence.")
    expected: str = Field(..., example="Quantum computing uses qubits to process complex data simultaneously.")
    actual: str = Field(..., example="Quantum computing leverages qubits for parallel processing of complex states.")

class BatchEvaluationRequest(BaseModel):
    test_cases: List[TestCaseSchema]

@app.post("/v1/evaluate/batch", summary="Run Full Enterprise Evaluation Suite")
def evaluate_batch(payload: BatchEvaluationRequest):
    try:
        payloads = [
            {"prompt": t.prompt, "expected": t.expected, "actual": t.actual} 
            for t in payload.test_cases
        ]
        df_metrics = evaluator.execute_evaluation_suite(payloads)
        
        passed_count = int((df_metrics["evaluation_verdict"] == "PASSED").sum())
        total_count = len(payloads)
        
        return {
            "status": "success",
            "summary": {
                "total_evaluated": total_count,
                "passed_evaluations": passed_count,
                "success_rate_pct": round((passed_count / total_count) * 100, 2) if total_count > 0 else 0.0
            },
            "detailed_metrics": df_metrics.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
