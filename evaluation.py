from dataclasses import dataclass
from datetime import datetime
from typing import List

@dataclass
class EvaluationResult:
    """Evaluation result for a resume against a job description"""
    resume_id: str
    job_id: str
    relevance_score: float
    hard_match_score: float
    semantic_match_score: float
    missing_skills: List[str]
    matched_skills: List[str]
    verdict: str  # HIGH, MEDIUM, LOW
    feedback: str
    suggestions: List[str]
    evaluated_at: datetime
