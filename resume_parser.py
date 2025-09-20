import re
import hashlib
from datetime import datetime
import spacy
import logging
from typing import Dict, List
from models.resume import Resume

logger = logging.getLogger(__name__)

# Load spaCy NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class ResumeParser:
    """Parse resume text into structured format"""

    def __init__(self):
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')

        self.section_headers = {
            'experience': ['experience', 'work history', 'employment', 'professional experience'],
            'education': ['education', 'academic', 'qualification', 'degree'],
            'skills': ['skills', 'technical skills', 'competencies', 'technologies'],
            'projects': ['projects', 'portfolio', 'achievements'],
            'certifications': ['certifications', 'certificates', 'training'],
            'summary': ['summary', 'objective', 'profile', 'about']
        }

    def parse(self, text: str, file_path: str) -> Resume:
        text = self._clean_text(text)

        name = self._extract_name(text)
        email = self._extract_email(text)
        phone = self._extract_phone(text)

        sections = self._extract_sections(text)
        skills = self._extract_skills(sections.get('skills', ''), text)
        experience = self._extract_experience(sections.get('experience', ''))
        education = self._extract_education(sections.get('education', ''))
        projects = self._extract_projects(sections.get('projects', ''))
        certifications = self._extract_certifications(sections.get('certifications', ''))
        summary = sections.get('summary', '')

        resume_id = hashlib.md5(f"{email}_{datetime.now()}".encode()).hexdigest()[:12]

        return Resume(
            id=resume_id,
            name=name,
            email=email,
            phone=phone,
            skills=skills,
            experience=experience,
            education=education,
            projects=projects,
            certifications=certifications,
            summary=summary,
            raw_text=text,
            file_path=file_path,
            created_at=datetime.now()
        )

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\-\.\,\@\+\(\)\/\:]', '', text)
        return text.strip()

    def _extract_name(self, text: str) -> str:
        doc = nlp(text[:500])
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text
        lines = text.split('\n')
        if lines:
            potential_name = lines[0].strip()
            if len(potential_name.split()) <= 4:
                return potential_name
        return "Unknown"

    def _extract_email(self, text: str) -> str:
        matches = self.email_pattern.findall(text)
        return matches[0] if matches else ""

    def _extract_phone(self, text: str) -> str:
        matches = self.phone_pattern.findall(text)
        return matches[0] if matches else ""

    def _extract_sections(self, text: str) -> Dict[str, str]:
        sections = {}
        lines = text.split('\n')
        current_section = None
        current_content = []

        for line in lines:
            line_lower = line.lower().strip()
            section_found = None
            for section_name, headers in self.section_headers.items():
                if any(header in line_lower for header in headers):
                    section_found = section_name
                    break

            if section_found:
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = section_found
                current_content = []
            else:
                current_content.append(line)

        if current_section:
            sections[current_section] = '\n'.join(current_content)
        return sections

    def _extract_skills(self, skills_section: str, full_text: str) -> List[str]:
        skills = []
        tech_skills = [
            'python', 'java', 'javascript', 'c++', 'sql', 'html', 'css', 'react',
            'angular', 'vue', 'node.js', 'django', 'flask', 'spring', 'docker',
            'kubernetes', 'aws', 'azure', 'gcp', 'git', 'jenkins', 'machine learning',
            'deep learning', 'nlp', 'computer vision', 'tensorflow', 'pytorch',
            'scikit-learn', 'pandas', 'numpy', 'mongodb', 'postgresql', 'mysql'
        ]
        if skills_section:
            potential_skills = re.split(r'[,;\|\n]', skills_section)
            skills.extend([s.strip() for s in potential_skills if s.strip()])
        full_text_lower = full_text.lower()
        for skill in tech_skills:
            if skill in full_text_lower:
                skills.append(skill)
        skills = list(set([s.lower().strip() for s in skills if len(s) > 1]))
        return skills

    def _extract_experience(self, experience_section: str) -> List[Dict]:
        experiences = []
        if not experience_section:
            return experiences
        lines = experience_section.split('\n')
        current_exp = {}
        for line in lines:
            line = line.strip()
            if not line:
                if current_exp:
                    experiences.append(current_exp)
                    current_exp = {}
                continue
            if not current_exp.get('title'):
                current_exp['title'] = line
            elif not current_exp.get('company'):
                current_exp['company'] = line
            else:
                current_exp['description'] = current_exp.get('description', '') + ' ' + line
        if current_exp:
            experiences.append(current_exp)
        return experiences

    def _extract_education(self, education_section: str) -> List[Dict]:
        education = []
        if not education_section:
            return education
        degree_patterns = ['bachelor', 'master', 'phd', 'b.tech', 'm.tech', 'mba', 'b.e', 'm.e']
        lines = education_section.split('\n')
        current_edu = {}
        for line in lines:
            line = line.strip()
            if not line:
                if current_edu:
                    education.append(current_edu)
                    current_edu = {}
                continue
            line_lower = line.lower()
            for degree in degree_patterns:
                if degree in line_lower:
                    current_edu['degree'] = line
                    break
            if 'degree' not in current_edu:
                current_edu['institution'] = line
        if current_edu:
            education.append(current_edu)
        return education

    def _extract_projects(self, projects_section: str) -> List[Dict]:
        projects = []
        if not projects_section:
            return projects
        lines = projects_section.split('\n')
        current_project = {}
        for line in lines:
            line = line.strip()
            if not line:
                if current_project:
                    projects.append(current_project)
                    current_project = {}
                continue
            if not current_project.get('name'):
                current_project['name'] = line
            else:
                current_project['description'] = current_project.get('description', '') + ' ' + line
        if current_project:
            projects.append(current_project)
        return projects

    def _extract_certifications(self, cert_section: str) -> List[str]:
        if not cert_section:
            return []
        certs = [line.strip() for line in cert_section.split('\n') if line.strip() and len(line) > 5]
        return certs
