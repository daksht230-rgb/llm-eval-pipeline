import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any

class AdvancedLLMEvaluator:
    def __init__(self, target_model: str = "gpt-4o-evaluator"):
        self.target_model = target_model

    def measure_latency(self, start_time: float) -> float:
        return round(time.time() - start_time, 4)

    def compute_semantic_alignment(self, expected: str, actual: str) -> float:
        """Computes Jaccard token similarity with weighted normalization."""
        exp_tokens = set(expected.lower().split())
        act_tokens = set(actual.lower().split())
        
        if not exp_tokens or not act_tokens:
            return 0.0
            
        intersection = exp_tokens.intersection(act_tokens)
        union = exp_tokens.union(act_tokens)
        
        score = (len(intersection) / len(union)) * 100.0
        return round(float(score), 2)

    def detect_hallucination_risk(self, prompt: str, actual: str) -> str:
        """Heuristic risk assessment for hallucinated or non-factual outputs."""
        risk_keywords = ["i am not sure", "as an AI", "however", "fabricate", "hypothetically"]
        matches = sum(1 for word in risk_keywords if word in actual.lower())
        if matches >= 2:
            return "HIGH_RISK"
        elif matches == 1:
            return "MODERATE_RISK"
        return "LOW_RISK"

    def execute_evaluation_suite(self, test_payloads: List[Dict[str, str]]) -> pd.DataFrame:
        results = []
        for idx, test in enumerate(test_payloads):
            t_start = time.time()
            
            prompt = test.get("prompt", "")
            expected = test.get("expected", "")
            actual = test.get("actual", "")
            
            latency = self.measure_latency(t_start)
            alignment_score = self.compute_semantic_alignment(expected, actual)
            hallucination_status = self.detect_hallucination_risk(prompt, actual)
            
            passed = alignment_score >= 70.0 and hallucination_status != "HIGH_RISK"
            
            results.append({
                "test_case_id": f"TC-PROD-{idx+1:03d}",
                "alignment_score_pct": alignment_score,
                "latency_seconds": latency,
                "hallucination_flag": hallucination_status,
                "evaluation_verdict": "PASSED" if passed else "FAILED_REVIEW"
            })
            
        return pd.DataFrame(results)
