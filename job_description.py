from dataclasses import dataclass
from datetime import datetime
from typing import List

@dataclass
class JobDescription:
    """Structured representation of a job description"""
    id: str
    title: str
    company: str
    must_have_skills: List[str]
    good_to_have_skills: List[str]
    experience_required: str
    education: List[str]
    description: str
    location: str
    created_at: datetime
