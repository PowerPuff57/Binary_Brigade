import re
import hashlib
from datetime import datetime
from typing import List
from models.job_description import JobDescription

class JobDescriptionParser:
    """Parse job descriptions into structured format"""

    def parse(self, text: str, company: str = "", location: str = "") -> JobDescription:
        text = self._clean_text(text)
        title = self._extract_title(text)
        must_have_skills = self._extract_must_have_skills(text)
        good_to_have_skills = self._extract_good_to_have_skills(text)
        experience = self._extract_experience_requirement(text)
        education = self._extract_education_requirement(text)
        jd_id = hashlib.md5(f"{company}_{title}_{datetime.now()}".encode()).hexdigest()[:12]
        return JobDescription(
            id=jd_id,
            title=title,
            company=company,
            must_have_skills=must_have_skills,
            good_to_have_skills=good_to_have_skills,
            experience_required=experience,
            education=education,
            description=text,
            location=location,
            created_at=datetime.now()
        )

    def _clean_text(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()

    def _extract_title(self, text: str) -> str:
        patterns = [
            r'position\s*:\s*([^\n]+)',
            r'job\s*title\s*:\s*([^\n]+)',
            r'role\s*:\s*([^\n]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        lines = text.split('\n')
        return lines[0].strip() if lines else "Unknown Position"

    def _extract_must_have_skills(self, text: str) -> List[str]:
        skills = []
        patterns = [
            r'required\s*skills?\s*:?\s*([^\n]+(?:\n[^\n]+)*)',
            r'must\s*have\s*:?\s*([^\n]+(?:\n[^\n]+)*)',
            r'mandatory\s*skills?\s*:?\s*([^\n]+(?:\n[^\n]+)*)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                skills_text = match.group(1)
                skills.extend(re.split(r'[,;\|\n•]', skills_text))
                break
        skills = [s.strip().lower() for s in skills if s.strip() and len(s.strip()) > 2]
        return list(set(skills))[:20] or self._extract_skills_from_description(text)

    def _extract_good_to_have_skills(self, text: str) -> List[str]:
        skills = []
        patterns = [
            r'good\s*to\s*have\s*:?\s*([^\n]+(?:\n[^\n]+)*)',
            r'preferred\s*skills?\s*:?\s*([^\n]+(?:\n[^\n]+)*)',
            r'nice\s*to\s*have\s*:?\s*([^\n]+(?:\n[^\n]+)*)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                skills_text = match.group(1)
                skills.extend(re.split(r'[,;\|\n•]', skills_text))
                break
        skills = [s.strip().lower() for s in skills if s.strip() and len(s.strip()) > 2]
        return list(set(skills))[:10]

    def _extract_experience_requirement(self, text: str) -> str:
        patterns = [
            r'(\d+[\+\-]?\d*)\s*years?\s*(?:of\s*)?experience',
            r'experience\s*:\s*(\d+[\+\-]?\d*)\s*years?',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1) + " years"
        return "Not specified"

    def _extract_education_requirement(self, text: str) -> List[str]:
        degrees = ['bachelor', 'master', 'phd', 'b.tech', 'm.tech', 'mba', 'b.e', 'm.e', 'bsc', 'msc', 'graduate']
        text_lower = text.lower()
        return [degree.upper() for degree in degrees if degree in text_lower]

    def _extract_skills_from_description(self, text: str) -> List[str]:
        tech_skills = [
            'python', 'java', 'javascript', 'c++', 'sql', 'html', 'css',
            'react', 'angular', 'vue', 'node.js', 'django', 'flask',
            'spring', 'docker', 'kubernetes', 'aws', 'azure', 'gcp',
            'git', 'jenkins', 'machine learning', 'deep learning',
            'data science', 'analytics', 'agile', 'scrum'
        ]
        text_lower = text.lower()
        return [skill for skill in tech_skills if skill in text_lower]
