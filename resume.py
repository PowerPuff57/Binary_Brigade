from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict

@dataclass
class Resume:
    """Structured representation of a resume"""
    id: str
    name: str
    email: str
    phone: str
    skills: List[str]
    experience: List[Dict]
    education: List[Dict]
    projects: List[Dict]
    certifications: List[str]
    summary: str
    raw_text: str
    file_path: str
    created_at: datetime
