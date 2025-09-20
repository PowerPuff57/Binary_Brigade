# Automated Resume Relevance Check System
# Complete implementation with all components
import os
import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

# Document Processing
try:
    import PyPDF2
    import pdfplumber
    import docx2txt
    from docx import Document
except ImportError:
    print("Warning: Some document processing libraries not found. Install with: pip install PyPDF2 pdfplumber docx2txt python-docx")

# NLP and Text Processing
try:
    import spacy
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from fuzzywuzzy import fuzz
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
except ImportError:
    print("Warning: Some NLP libraries not found. Install with: pip install spacy nltk fuzzywuzzy scikit-learn numpy")

# Web Framework
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import sqlite3
from werkzeug.utils import secure_filename

# Initialize components
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Warning: spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

# Data Classes
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

class DocumentProcessor:
    """Handles document extraction and parsing"""

    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract text from PDF using multiple methods for robustness"""
        text = ""
        
        # Try pdfplumber first (better for tables)
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}")
            
        # Fallback to PyPDF2
        if not text.strip():
            try:
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
            except Exception as e:
                logger.error(f"PDF extraction failed: {e}")
                
        return text.strip()

    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        """Extract text from DOCX files"""
        try:
            # Try docx2txt first (simpler)
            text = docx2txt.process(file_path)
            if not text:
                # Fallback to python-docx
                doc = Document(file_path)
                text = "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            text = ""
            
        return text.strip()

    @staticmethod
    def extract_text(file_path: str) -> str:
        """Extract text from any supported document format"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return DocumentProcessor.extract_text_from_pdf(file_path)
        elif file_ext in ['.docx', '.doc']:
            return DocumentProcessor.extract_text_from_docx(file_path)
        elif file_ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

class ResumeParser:
    """Parse resume text into structured format"""
    
    def __init__(self):
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
        
        # Common section headers
        self.section_headers = {
            'experience': ['experience', 'work history', 'employment', 'professional experience'],
            'education': ['education', 'academic', 'qualification', 'degree'],
            'skills': ['skills', 'technical skills', 'competencies', 'technologies'],
            'projects': ['projects', 'portfolio', 'achievements'],
            'certifications': ['certifications', 'certificates', 'training'],
            'summary': ['summary', 'objective', 'profile', 'about']
        }

    def parse(self, text: str, file_path: str) -> Resume:
        """Parse resume text into structured Resume object"""
        # Clean text
        text = self._clean_text(text)
        
        # Extract basic info
        name = self._extract_name(text)
        email = self._extract_email(text)
        phone = self._extract_phone(text)
        
        # Extract sections
        sections = self._extract_sections(text)
        
        # Parse specific sections
        skills = self._extract_skills(sections.get('skills', ''), text)
        experience = self._extract_experience(sections.get('experience', ''))
        education = self._extract_education(sections.get('education', ''))
        projects = self._extract_projects(sections.get('projects', ''))
        certifications = self._extract_certifications(sections.get('certifications', ''))
        summary = sections.get('summary', '')
        
        # Generate unique ID
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
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\-\.\,\@\+\(\)\/\:]', '', text)
        return text.strip()

    def _extract_name(self, text: str) -> str:
        """Extract candidate name using NLP"""
        if nlp:
            doc = nlp(text[:500])
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    return ent.text
        
        # Fallback: first line often contains name
        lines = text.split('\n')
        if lines:
            potential_name = lines[0].strip()
            if len(potential_name.split()) <= 4:
                return potential_name
        
        return "Unknown"

    def _extract_email(self, text: str) -> str:
        """Extract email address"""
        matches = self.email_pattern.findall(text)
        return matches[0] if matches else ""

    def _extract_phone(self, text: str) -> str:
        """Extract phone number"""
        matches = self.phone_pattern.findall(text)
        return matches[0] if matches else ""

    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Split resume into sections based on headers"""
        sections = {}
        lines = text.split('\n')
        
        current_section = None
        current_content = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if line is a section header
            section_found = None
            for section_name, headers in self.section_headers.items():
                if any(header in line_lower for header in headers):
                    section_found = section_name
                    break
            
            if section_found:
                # Save previous section
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                
                current_section = section_found
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_section:
            sections[current_section] = '\n'.join(current_content)
        
        return sections

    def _extract_skills(self, skills_section: str, full_text: str) -> List[str]:
        """Extract skills from resume"""
        skills = []
        
        # Common technical skills to look for
        tech_skills = [
            'python', 'java', 'javascript', 'c++', 'sql', 'html', 'css', 'react',
            'angular', 'vue', 'node.js', 'django', 'flask', 'spring', 'docker',
            'kubernetes', 'aws', 'azure', 'gcp', 'git', 'jenkins', 'machine learning',
            'deep learning', 'nlp', 'computer vision', 'tensorflow', 'pytorch',
            'scikit-learn', 'pandas', 'numpy', 'mongodb', 'postgresql', 'mysql'
        ]
        
        # Extract from skills section
        if skills_section:
            potential_skills = re.split(r'[,;\|\n]', skills_section)
            skills.extend([s.strip() for s in potential_skills if s.strip()])
        
        # Also search full text for tech skills
        full_text_lower = full_text.lower()
        for skill in tech_skills:
            if skill in full_text_lower:
                skills.append(skill)
        
        # Remove duplicates and clean
        skills = list(set([s.lower().strip() for s in skills if len(s) > 1]))
        
        return skills

    def _extract_experience(self, experience_section: str) -> List[Dict]:
        """Extract work experience"""
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
                if 'description' not in current_exp:
                    current_exp['description'] = line
                else:
                    current_exp['description'] += ' ' + line
        
        if current_exp:
            experiences.append(current_exp)
        
        return experiences

    def _extract_education(self, education_section: str) -> List[Dict]:
        """Extract education details"""
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
        """Extract projects"""
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
                if 'description' not in current_project:
                    current_project['description'] = line
                else:
                    current_project['description'] += ' ' + line
        
        if current_project:
            projects.append(current_project)
        
        return projects

    def _extract_certifications(self, cert_section: str) -> List[str]:
        """Extract certifications"""
        if not cert_section:
            return []
        
        certs = []
        lines = cert_section.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 5:
                certs.append(line)
        
        return certs

class JobDescriptionParser:
    """Parse job descriptions into structured format"""
    
    def parse(self, text: str, company: str = "", location: str = "") -> JobDescription:
        """Parse JD text into structured JobDescription object"""
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
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _extract_title(self, text: str) -> str:
        """Extract job title"""
        # First try explicit title patterns
        title_patterns = [
            r'job\s*title\s*:?\s*([^\n]+)',
            r'position\s*:?\s*([^\n]+)',
            r'role\s*:?\s*([^\n]+)',
            r'designation\s*:?\s*([^\n]+)',
            r'vacancy\s*:?\s*([^\n]+)',
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                # Clean up the title
                title = re.sub(r'[^\w\s\-\(\)\/]', '', title)
                if len(title) <= 100:  # Reasonable title length
                    return title

        # Try to extract from first few lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Common job title keywords to look for
        job_keywords = [
            'developer', 'engineer', 'analyst', 'manager', 'lead', 'senior', 'junior',
            'intern', 'associate', 'specialist', 'consultant', 'architect', 'designer',
            'scientist', 'researcher', 'coordinator', 'executive', 'director', 'head',
            'officer', 'administrator', 'technician', 'programmer', 'tester'
        ]
        
        # Look through first 5 lines for potential job titles
        for line in lines[:5]:
            line_lower = line.lower()
            
            # Skip lines that are too long (likely descriptions)
            if len(line) > 100:
                continue
                
            # Skip lines with common non-title indicators
            skip_indicators = [
                'company', 'description', 'about', 'we are', 'location', 'salary',
                'responsibilities', 'requirements', 'qualifications', 'experience',
                'skills', 'benefits', 'apply', 'contact', 'email', 'phone'
            ]
            
            if any(indicator in line_lower for indicator in skip_indicators):
                continue
                
            # Check if line contains job-related keywords
            if any(keyword in line_lower for keyword in job_keywords):
                # Clean the line
                cleaned_title = re.sub(r'[^\w\s\-\(\)\/]', '', line)
                if len(cleaned_title.strip()) > 0:
                    return cleaned_title.strip()
        
        # Final fallback: use first short line
        for line in lines[:3]:
            if len(line) <= 60 and len(line) >= 5:
                cleaned_title = re.sub(r'[^\w\s\-\(\)\/]', '', line)
                return cleaned_title.strip()

        return "Software Position"  # Default fallback

    def _extract_must_have_skills(self, text: str) -> List[str]:
        """Extract required skills"""
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
        if not skills:
            skills = self._extract_skills_from_description(text)
        return list(set(skills))[:20]

    def _extract_good_to_have_skills(self, text: str) -> List[str]:
        """Extract nice-to-have skills"""
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
        """Extract experience requirement"""
        patterns = [
            r'(\d+[\+\-]?\d*)\s*years?\s*(?:of\s*)?experience',
            r'experience\s*:\s*(\d+[\+\-]?\d*)\s*years?',
        ]
        for pattern in patterns:
            match = re.search(pattern, re.sub(r'\s+', ' ', text), re.IGNORECASE)
            if match:
                return match.group(1) + " years"
        return "Not specified"

    def _extract_education_requirement(self, text: str) -> List[str]:
        """Extract education requirements"""
        education = []
        degree_patterns = [
            'bachelor', 'master', 'phd', 'b.tech', 'm.tech',
            'mba', 'b.e', 'm.e', 'bsc', 'msc', 'graduate'
        ]
        text_lower = text.lower()
        for degree in degree_patterns:
            if degree in text_lower:
                education.append(degree.upper())
        return education

    def _extract_skills_from_description(self, text: str) -> List[str]:
        """Extract skills from job description using NLP"""
        tech_skills = [
            'python', 'java', 'javascript', 'c++', 'sql', 'html', 'css',
            'react', 'angular', 'vue', 'node.js', 'django', 'flask',
            'spring', 'docker', 'kubernetes', 'aws', 'azure', 'gcp',
            'git', 'jenkins', 'machine learning', 'deep learning',
            'data science', 'analytics', 'agile', 'scrum'
        ]
        skills = []
        text_lower = text.lower()
        for skill in tech_skills:
            if skill in text_lower:
                skills.append(skill)
        return skills
class RelevanceEvaluator:
    """Evaluate resume relevance against job descriptions"""
    
    def __init__(self):
        pass

    def evaluate(self, resume: Resume, job_desc: JobDescription) -> EvaluationResult:
        """Evaluate resume against job description"""
        
        # Hard match scoring
        hard_match_result = self._hard_match_scoring(resume, job_desc)
        
        # For now, semantic match is disabled
        semantic_match_result = {'score': 0, 'explanation': 'Semantic matching disabled'}
        
        # Combine scores
        hard_score = hard_match_result['score']
        semantic_score = semantic_match_result['score']
        
        final_score = (hard_score * 0.8 + semantic_score * 0.2)
        
        verdict = self._determine_verdict(final_score)
        
        feedback = self._generate_feedback(
            resume, job_desc, hard_match_result, semantic_match_result, final_score
        )
        
        suggestions = self._generate_suggestions(
            resume, job_desc, hard_match_result['missing_skills']
        )
        
        return EvaluationResult(
            resume_id=resume.id,
            job_id=job_desc.id,
            relevance_score=round(final_score, 2),
            hard_match_score=round(hard_score, 2),
            semantic_match_score=round(semantic_score, 2),
            missing_skills=hard_match_result['missing_skills'],
            matched_skills=hard_match_result['matched_skills'],
            verdict=verdict,
            feedback=feedback,
            suggestions=suggestions,
            evaluated_at=datetime.now()
        )

    def _hard_match_scoring(self, resume: Resume, job_desc: JobDescription) -> Dict:
        """Perform hard match scoring based on keywords and skills"""
        
        score = 0
        matched_skills = []
        missing_skills = []
        
        # Skill matching (40 points)
        resume_skills = set([s.lower() for s in resume.skills])
        must_have_skills = set([s.lower() for s in job_desc.must_have_skills])
        good_to_have_skills = set([s.lower() for s in job_desc.good_to_have_skills])
        
        # Must-have skills matching
        if must_have_skills:
            matched_must_have = resume_skills.intersection(must_have_skills)
            matched_skills.extend(list(matched_must_have))
            score += (len(matched_must_have) / len(must_have_skills)) * 30
            missing_must_have = must_have_skills - matched_must_have
            missing_skills.extend(list(missing_must_have))
        
        # Good-to-have skills matching
        if good_to_have_skills:
            matched_good_to_have = resume_skills.intersection(good_to_have_skills)
            matched_skills.extend(list(matched_good_to_have))
            score += (len(matched_good_to_have) / len(good_to_have_skills)) * 10
        
        # Education matching (15 points)
        if job_desc.education:
            resume_education_text = ' '.join([
                str(edu.get('degree', '')) for edu in resume.education
            ]).lower()
            education_match = False
            for req_edu in job_desc.education:
                if req_edu.lower() in resume_education_text:
                    education_match = True
                    break
            if education_match:
                score += 15
        
        # Experience matching (15 points)
        if job_desc.experience_required != "Not specified":
            years_match = re.search(r'(\d+)', job_desc.experience_required)
            if years_match:
                required_years = int(years_match.group(1))
                candidate_years = len(resume.experience) * 2
                if candidate_years >= required_years:
                    score += 15
                elif candidate_years >= required_years * 0.7:
                    score += 10
                elif candidate_years >= required_years * 0.5:
                    score += 5
        
        # Project relevance (20 points)
        if resume.projects:
            project_text = ' '.join([
                f"{p.get('name', '')} {p.get('description', '')}" for p in resume.projects
            ]).lower()
            relevant_project_keywords = must_have_skills.union(good_to_have_skills)
            project_relevance = sum(
                1 for keyword in relevant_project_keywords if keyword in project_text
            )
            if project_relevance > 0:
                score += min(20, project_relevance * 3)
        
        # Certification relevance (20 points)
        if resume.certifications:
            cert_text = ' '.join(resume.certifications).lower()
            cert_relevance = sum(
                1 for skill in must_have_skills.union(good_to_have_skills)
                if skill in cert_text
            )
            if cert_relevance > 0:
                score += min(20, cert_relevance * 5)
        
        return {
            'score': min(100, score),
            'matched_skills': list(set(matched_skills)),
            'missing_skills': list(set(missing_skills))
        }

    def _determine_verdict(self, score: float) -> str:
        """Determine verdict based on score"""
        if score >= 75:
            return "HIGH"
        elif score >= 50:
            return "MEDIUM"
        else:
            return "LOW"

    def _generate_feedback(self, resume: Resume, job_desc: JobDescription, hard_match: Dict, semantic_match: Dict, final_score: float) -> str:
        """Generate detailed feedback"""
        feedback_parts = []
        
        feedback_parts.append(f"Overall Relevance Score: {final_score:.1f}/100")
        
        if hard_match['matched_skills']:
            feedback_parts.append(f"Matched Skills: {', '.join(hard_match['matched_skills'][:5])}")
        
        if hard_match['missing_skills']:
            feedback_parts.append(f"Missing Key Skills: {', '.join(hard_match['missing_skills'][:5])}")
        
        if job_desc.experience_required != "Not specified":
            exp_years = len(resume.experience) * 2
            feedback_parts.append(f"Estimated Experience: ~{exp_years} years (Required: {job_desc.experience_required})")
        
        if resume.projects:
            feedback_parts.append(f"Projects Found: {len(resume.projects)} project(s)")
        else:
            feedback_parts.append("No projects mentioned - consider adding relevant projects")
        
        return " | ".join(feedback_parts)

    def _generate_suggestions(self, resume: Resume, job_desc: JobDescription, missing_skills: List[str]) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        
        if missing_skills:
            top_missing = missing_skills[:3]
            suggestions.append(f"Acquire these critical skills: {', '.join(top_missing)}")
        
        if not resume.projects or len(resume.projects) < 2:
            suggestions.append("Add 2-3 relevant projects demonstrating practical experience")
        
        if not resume.certifications:
            suggestions.append("Consider obtaining relevant certifications in your domain")
        
        if len(resume.experience) < 2:
            suggestions.append("Gain more practical experience through internships or freelance work")
        
        if not resume.summary or len(resume.summary) < 50:
            suggestions.append("Add a compelling professional summary highlighting your key strengths")
        
        return suggestions[:5]

class DatabaseManager:
    """Manage SQLite database for storing evaluations"""
    
    def __init__(self, db_path: str = "resume_evaluation.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resumes (
                id TEXT PRIMARY KEY,
                name TEXT,
                email TEXT,
                phone TEXT,
                skills TEXT,
                experience TEXT,
                education TEXT,
                projects TEXT,
                certifications TEXT,
                summary TEXT,
                file_path TEXT,
                created_at TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS job_descriptions (
                id TEXT PRIMARY KEY,
                title TEXT,
                company TEXT,
                must_have_skills TEXT,
                good_to_have_skills TEXT,
                experience_required TEXT,
                education TEXT,
                description TEXT,
                location TEXT,
                created_at TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_id TEXT,
                job_id TEXT,
                relevance_score REAL,
                hard_match_score REAL,
                semantic_match_score REAL,
                missing_skills TEXT,
                matched_skills TEXT,
                verdict TEXT,
                feedback TEXT,
                suggestions TEXT,
                evaluated_at TEXT,
                FOREIGN KEY (resume_id) REFERENCES resumes (id),
                FOREIGN KEY (job_id) REFERENCES job_descriptions (id)
            )
        ''')
        
        conn.commit()
        conn.close()

    def save_resume(self, resume: Resume):
        """Save resume to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO resumes 
            (id, name, email, phone, skills, experience, education, projects, certifications, summary, file_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            resume.id,
            resume.name,
            resume.email,
            resume.phone,
            json.dumps(resume.skills),
            json.dumps(resume.experience),
            json.dumps(resume.education),
            json.dumps(resume.projects),
            json.dumps(resume.certifications),
            resume.summary,
            resume.file_path,
            resume.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()

    def save_job_description(self, job_desc: JobDescription):
        """Save job description to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO job_descriptions 
            (id, title, company, must_have_skills, good_to_have_skills, experience_required, education, description, location, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            job_desc.id,
            job_desc.title,
            job_desc.company,
            json.dumps(job_desc.must_have_skills),
            json.dumps(job_desc.good_to_have_skills),
            job_desc.experience_required,
            json.dumps(job_desc.education),
            job_desc.description,
            job_desc.location,
            job_desc.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()

    def save_evaluation(self, evaluation: EvaluationResult):
        """Save evaluation result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO evaluations 
            (resume_id, job_id, relevance_score, hard_match_score, semantic_match_score, missing_skills, matched_skills, verdict, feedback, suggestions, evaluated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            evaluation.resume_id,
            evaluation.job_id,
            evaluation.relevance_score,
            evaluation.hard_match_score,
            evaluation.semantic_match_score,
            json.dumps(evaluation.missing_skills),
            json.dumps(evaluation.matched_skills),
            evaluation.verdict,
            evaluation.feedback,
            json.dumps(evaluation.suggestions),
            evaluation.evaluated_at.isoformat()
        ))
        
        conn.commit()
        conn.close()

    def get_evaluations_for_job(self, job_id: str) -> List[Dict]:
        """Get all evaluations for a specific job"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT e.*, r.name, r.email
            FROM evaluations e
            JOIN resumes r ON e.resume_id = r.id
            WHERE e.job_id = ?
            ORDER BY e.relevance_score DESC
        ''', (job_id,))
        
        columns = [description[0] for description in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            result = dict(zip(columns, row))
            # Parse JSON fields
            result['missing_skills'] = json.loads(result['missing_skills'])
            result['matched_skills'] = json.loads(result['matched_skills'])
            result['suggestions'] = json.loads(result['suggestions'])
            results.append(result)
        
        conn.close()
        return results

    def get_all_jobs(self) -> List[Dict]:
        """Get all job descriptions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM job_descriptions ORDER BY created_at DESC')
        columns = [description[0] for description in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            result = dict(zip(columns, row))
            result['must_have_skills'] = json.loads(result['must_have_skills'])
            result['good_to_have_skills'] = json.loads(result['good_to_have_skills'])
            result['education'] = json.loads(result['education'])
            results.append(result)
        
        conn.close()
        return results

# Flask Web Application
app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize components
db_manager = DatabaseManager()
resume_parser = ResumeParser()
jd_parser = JobDescriptionParser()
evaluator = RelevanceEvaluator()

# HTML Template
DASHBOARD_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Resume Relevance Check System</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            color: #666;
            font-size: 1.1em;
        }
        .card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }
        .tab {
            padding: 12px 24px;
            background: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .tab.active {
            background: #667eea;
            color: white;
        }
        .tab:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .upload-section {
            display: none;
        }
        .upload-section.active {
            display: block;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        input, textarea, select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 5px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        input:focus, textarea:focus, select:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        .file-input {
            padding: 10px;
            background: #f5f5f5;
        }
        .btn {
            background: #667eea;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s;
            font-weight: 600;
        }
        .btn:hover {
            background: #5a67d8;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        .results {
            margin-top: 30px;
        }
        .result-item {
            background: #f9f9f9;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
        }
        .score {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .high {
            color: #10b981;
        }
        .medium {
            color: #f59e0b;
        }
        .low {
            color: #ef4444;
        }
        .skills-list {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }
        .skill-tag {
            background: #e0e7ff;
            color: #4338ca;
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 12px;
        }
        .missing-skill {
            background: #fee2e2;
            color: #dc2626;
        }
        .suggestion {
            background: #fef3c7;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            font-size: 14px;
        }
        .loading {
            text-align: center;
            padding: 20px;
        }
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% {
                transform: rotate(0deg);
            }
            100% {
                transform: rotate(360deg);
            }
        }
        .alert {
            padding: 15px;
            margin-bottom: 20px;
            border: 1px solid transparent;
            border-radius: 4px;
        }
        .alert-success {
            color: #155724;
            background-color: #d4edda;
            border-color: #c3e6cb;
        }
        .alert-error {
            color: #721c24;
            background-color: #f8d7da;
            border-color: #f5c6cb;
        }
        .stats {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .stat-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            flex: 1;
            min-width: 150px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .stat-number {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            color: #666;
            font-size: 14px;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Resume Relevance Check System</h1>
            <p class="subtitle">AI-Powered Resume Evaluation for Innomatics Research Labs</p>
        </div>
        
        <div class="tabs">
            <button class="tab active" onclick="switchTab(event, 'jd')">Upload Job Description</button>
            <button class="tab" onclick="switchTab(event, 'resume')">Upload Resume</button>
            <button class="tab" onclick="switchTab(event, 'results')">View Results</button>
        </div>

        <div class="card">
            <div id="jd-section" class="upload-section active">
                <h2>📋 Upload Job Description</h2>
                <form id="jd-form">
                    <div class="form-group">
                        <label>Company Name *</label>
                        <input type="text" name="company" required placeholder="e.g., Innomatics Research Labs">
                    </div>
                    <div class="form-group">
                        <label>Location</label>
                        <input type="text" name="location" placeholder="e.g., Bangalore, Hyderabad">
                    </div>
                    <div class="form-group">
                        <label>Job Description (Text)</label>
                        <textarea name="jd_text" rows="10" placeholder="Paste job description here..."></textarea>
                    </div>
                    <div class="form-group">
                        <label>Or Upload JD File</label>
                        <input type="file" name="jd_file" class="file-input" accept=".pdf,.docx,.txt">
                    </div>
                    <button type="submit" class="btn">Upload Job Description</button>
                </form>
            </div>

            <div id="resume-section" class="upload-section">
                <h2>📄 Upload Resume for Evaluation</h2>
                <form id="resume-form">
                    <div class="form-group">
                        <label>Select Job *</label>
                        <select name="job_id" id="job-select" required>
                            <option value="">Select a job...</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Upload Resume *</label>
                        <input type="file" name="resume_file" class="file-input" accept=".pdf,.docx,.txt" required>
                    </div>
                    <button type="submit" class="btn">Evaluate Resume</button>
                </form>
            </div>

            <div id="results-section" class="upload-section">
                <h2>📊 Evaluation Results</h2>
                <div class="form-group">
                    <label>Select Job to View Results</label>
                    <select id="results-job-select">
                        <option value="">Select a job...</option>
                    </select>
                </div>
                <div id="results-stats" class="stats" style="display: none;">
                    <div class="stat-card">
                        <div class="stat-number" id="total-resumes">0</div>
                        <div class="stat-label">Total Resumes</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="high-matches">0</div>
                        <div class="stat-label">High Matches</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="medium-matches">0</div>
                        <div class="stat-label">Medium Matches</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="low-matches">0</div>
                        <div class="stat-label">Low Matches</div>
                    </div>
                </div>
                <div id="results-container"></div>
            </div>
        </div>

        <div id="loading" class="card loading" style="display: none;">
            <div class="spinner"></div>
            <p style="margin-top: 15px;">Processing...</p>
        </div>

        <div id="alert-container"></div>
    </div>

    <script>
        let currentJobs = [];

        function showAlert(message, type = 'success') {
            const alertContainer = document.getElementById('alert-container');
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${type}`;
            alertDiv.textContent = message;
            alertContainer.appendChild(alertDiv);
            
            setTimeout(() => {
                alertDiv.remove();
            }, 5000);
        }

        function switchTab(event, tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.upload-section').forEach(s => s.classList.remove('active'));
            
            event.target.classList.add('active');
            document.getElementById(tab + '-section').classList.add('active');
            
            if (tab === 'resume' || tab === 'results') {
                loadJobs();
            }
        }

        async function loadJobs() {
            try {
                const response = await fetch('/api/jobs');
                const jobs = await response.json();
                currentJobs = jobs;

                const jobSelect = document.getElementById('job-select');
                const resultsJobSelect = document.getElementById('results-job-select');

                jobSelect.innerHTML = '<option value="">Select a job...</option>';
                resultsJobSelect.innerHTML = '<option value="">Select a job...</option>';

                jobs.forEach(job => {
                    const option = document.createElement('option');
                    option.value = job.id;
                    option.textContent = `${job.title} at ${job.company}`;
                    jobSelect.appendChild(option);
                    resultsJobSelect.appendChild(option.cloneNode(true));
                });
            } catch (error) {
                console.error('Error loading jobs:', error);
                showAlert('Error loading jobs', 'error');
            }
        }

        async function handleFormSubmit(event) {
            event.preventDefault();
            showLoading();
            
            const form = event.target;
            const formData = new FormData(form);
            const isJdForm = form.id === 'jd-form';
            const endpoint = isJdForm ? '/api/upload_jd' : '/api/upload_resume';
            
            try {
                const response = await fetch(endpoint, {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.status === 'success') {
                    showAlert(result.message || 'File processed successfully!');
                    form.reset();
                    
                    if (isJdForm) {
                        loadJobs();
                    } else {
                        // Switch to results tab and show the evaluation
                        switchTab({target: document.querySelectorAll('.tab')[2]}, 'results');
                        document.getElementById('results-job-select').value = formData.get('job_id');
                        handleResultsSelect({target: {value: formData.get('job_id')}});
                    }
                } else {
                    showAlert('Error: ' + result.message, 'error');
                }
            } catch (error) {
                console.error('Form submission error:', error);
                showAlert('An error occurred: ' + error.message, 'error');
            } finally {
                hideLoading();
            }
        }

        async function handleResultsSelect(event) {
            const jobId = event.target.value;
            const container = document.getElementById('results-container');
            const statsDiv = document.getElementById('results-stats');
            
            container.innerHTML = '';
            statsDiv.style.display = 'none';
            
            if (!jobId) {
                return;
            }
            
            showLoading();
            
            try {
                const response = await fetch(`/api/evaluations/${jobId}`);
                const evaluations = await response.json();
                
                if (evaluations.length === 0) {
                    container.innerHTML = '<p style="text-align: center; color: #666; font-style: italic;">No resumes evaluated for this job yet.</p>';
                    return;
                }
                
                // Update stats
                const stats = {
                    total: evaluations.length,
                    high: evaluations.filter(e => e.verdict === 'HIGH').length,
                    medium: evaluations.filter(e => e.verdict === 'MEDIUM').length,
                    low: evaluations.filter(e => e.verdict === 'LOW').length
                };
                
                document.getElementById('total-resumes').textContent = stats.total;
                document.getElementById('high-matches').textContent = stats.high;
                document.getElementById('medium-matches').textContent = stats.medium;
                document.getElementById('low-matches').textContent = stats.low;
                statsDiv.style.display = 'flex';
                
                // Display evaluations
                evaluations.forEach(eval => {
                    const verdictClass = eval.verdict.toLowerCase();
                    const matchedSkillsHtml = eval.matched_skills.map(s => 
                        `<span class="skill-tag">${s}</span>`
                    ).join('');
                    const missingSkillsHtml = eval.missing_skills.map(s => 
                        `<span class="skill-tag missing-skill">${s}</span>`
                    ).join('');
                    const suggestionsHtml = eval.suggestions.map(s => 
                        `<div class="suggestion">💡 ${s}</div>`
                    ).join('');
                    
                    const resultHtml = `
                        <div class="result-item">
                            <h3>👤 ${eval.name} (${eval.email})</h3>
                            <p><strong>Verdict:</strong> <span class="score ${verdictClass}">${eval.verdict} (${eval.relevance_score}%)</span></p>
                            <p><strong>Hard Match Score:</strong> ${eval.hard_match_score}%</p>
                            <p><strong>Feedback:</strong> ${eval.feedback}</p>
                            ${matchedSkillsHtml ? `
                                <p style="margin-top: 15px;"><strong>✅ Matched Skills:</strong></p>
                                <div class="skills-list">${matchedSkillsHtml}</div>
                            ` : ''}
                            ${missingSkillsHtml ? `
                                <p style="margin-top: 15px;"><strong>❌ Missing Skills:</strong></p>
                                <div class="skills-list">${missingSkillsHtml}</div>
                            ` : ''}
                            ${suggestionsHtml ? `
                                <p style="margin-top: 15px;"><strong>📈 Suggestions for Improvement:</strong></p>
                                ${suggestionsHtml}
                            ` : ''}
                        </div>
                    `;
                    container.innerHTML += resultHtml;
                });
            } catch (error) {
                console.error('Error fetching results:', error);
                container.innerHTML = `<p style="color: red;">Error fetching results: ${error.message}</p>`;
            } finally {
                hideLoading();
            }
        }

        function showLoading() {
            document.getElementById('loading').style.display = 'block';
            document.querySelectorAll('.btn').forEach(btn => btn.disabled = true);
        }

        function hideLoading() {
            document.getElementById('loading').style.display = 'none';
            document.querySelectorAll('.btn').forEach(btn => btn.disabled = false);
        }

        // Event listeners
        document.addEventListener('DOMContentLoaded', () => {
            loadJobs();
            document.getElementById('jd-form').addEventListener('submit', handleFormSubmit);
            document.getElementById('resume-form').addEventListener('submit', handleFormSubmit);
            document.getElementById('results-job-select').addEventListener('change', handleResultsSelect);
        });
    </script>
</body>
</html>
'''

@app.route('/', methods=['GET'])
def index():
    """Serve the main dashboard HTML page"""
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/upload_jd', methods=['POST'])
def upload_jd():
    """Handle job description uploads"""
    try:
        # Check for file upload
        if 'jd_file' in request.files and request.files['jd_file'].filename != '':
            jd_file = request.files['jd_file']
            filename = secure_filename(jd_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            jd_file.save(file_path)
            jd_text = DocumentProcessor.extract_text(file_path)
        else:
            jd_text = request.form.get('jd_text')

        if not jd_text or jd_text.strip() == '':
            return jsonify({
                'status': 'error', 
                'message': 'No job description provided. Please enter text or upload a file.'
            }), 400

        company = request.form.get('company', 'Not specified')
        location = request.form.get('location', 'Not specified')
        
        job_desc = jd_parser.parse(jd_text, company, location)
        db_manager.save_job_description(job_desc)
        
        logger.info(f"Job description uploaded: {job_desc.title} at {job_desc.company}")
        
        return jsonify({
            'status': 'success',
            'message': f"Job description '{job_desc.title}' uploaded successfully!",
            'job_id': job_desc.id,
            'data': {
                'title': job_desc.title,
                'company': job_desc.company,
                'must_have_skills': job_desc.must_have_skills,
                'good_to_have_skills': job_desc.good_to_have_skills
            }
        })

    except Exception as e:
        logger.error(f"Error processing job description: {str(e)}")
        return jsonify({
            'status': 'error', 
            'message': f'Error processing job description: {str(e)}'
        }), 500

@app.route('/api/upload_resume', methods=['POST'])
def upload_resume():
    """Handle resume uploads and evaluation"""
    try:
        # Validate file upload
        if 'resume_file' not in request.files:
            return jsonify({
                'status': 'error', 
                'message': 'No resume file uploaded'
            }), 400

        resume_file = request.files['resume_file']
        if resume_file.filename == '':
            return jsonify({
                'status': 'error', 
                'message': 'No file selected'
            }), 400

        job_id = request.form.get('job_id')
        if not job_id:
            return jsonify({
                'status': 'error', 
                'message': 'No job selected'
            }), 400

        # Save uploaded file
        filename = secure_filename(resume_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        resume_file.save(file_path)

        # Extract text from resume
        resume_text = DocumentProcessor.extract_text(file_path)
        if not resume_text:
            return jsonify({
                'status': 'error', 
                'message': 'Could not extract text from resume file'
            }), 400

        # Parse resume
        resume = resume_parser.parse(resume_text, file_path)
        db_manager.save_resume(resume)

        # Get job description from database
        jobs_list = db_manager.get_all_jobs()
        job_data = None
        for job in jobs_list:
            if job['id'] == job_id:
                job_data = job
                break
        
        if not job_data:
            return jsonify({
                'status': 'error', 
                'message': 'Selected job not found'
            }), 404

        # Create JobDescription object
        try:
            created_at = datetime.fromisoformat(job_data['created_at'])
        except (ValueError, TypeError):
            created_at = datetime.now()

        job_desc = JobDescription(
            id=job_data['id'],
            title=job_data['title'],
            company=job_data['company'],
            must_have_skills=job_data['must_have_skills'],
            good_to_have_skills=job_data['good_to_have_skills'],
            experience_required=job_data['experience_required'],
            education=job_data['education'],
            description=job_data['description'],
            location=job_data['location'],
            created_at=created_at
        )

        # Evaluate resume
        evaluation = evaluator.evaluate(resume, job_desc)
        db_manager.save_evaluation(evaluation)

        logger.info(f"Resume evaluated: {resume.name} for {job_desc.title} - Score: {evaluation.relevance_score}")

        return jsonify({
            'status': 'success',
            'message': f'Resume evaluated successfully! Score: {evaluation.relevance_score}%',
            'evaluation': {
                **asdict(evaluation),
                'name': resume.name,
                'email': resume.email
            }
        })

    except Exception as e:
        logger.error(f"Error evaluating resume: {str(e)}")
        return jsonify({
            'status': 'error', 
            'message': f'Error evaluating resume: {str(e)}'
        }), 500

@app.route('/api/jobs', methods=['GET'])
def get_jobs():
    """Get all job descriptions"""
    try:
        jobs = db_manager.get_all_jobs()
        return jsonify(jobs)
    except Exception as e:
        logger.error(f"Error fetching jobs: {str(e)}")
        return jsonify([])

@app.route('/api/evaluations/<job_id>', methods=['GET'])
def get_evaluations(job_id):
    """Get evaluations for a specific job"""
    try:
        evaluations = db_manager.get_evaluations_for_job(job_id)
        return jsonify(evaluations)
    except Exception as e:
        logger.error(f"Error fetching evaluations: {str(e)}")
        return jsonify([])

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Resume Relevance Check System is running'})

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'status': 'error', 'message': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting Resume Relevance Check System...")
    print("🌐 Server will be available at: http://localhost:5000")
    print("📊 Dashboard available at: http://localhost:5000")
    print("=" * 60)
    
    # Check if required directories exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
        print(f"✅ Created upload directory: {app.config['UPLOAD_FOLDER']}")
    
    # Initialize database
    try:
        db_manager._init_database()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
    
    # Check if required libraries are available
    missing_libs = []
    try:
        import PyPDF2
        import pdfplumber
    except ImportError:
        missing_libs.append("PDF processing (pip install PyPDF2 pdfplumber)")
    
    try:
        import docx2txt
        from docx import Document
    except ImportError:
        missing_libs.append("DOCX processing (pip install docx2txt python-docx)")
    
    if missing_libs:
        print("⚠️  Optional libraries missing:")
        for lib in missing_libs:
            print(f"   - {lib}")
        print("   System will work with limited functionality")
    
    print("🔥 System ready! Upload job descriptions and evaluate resumes.")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)