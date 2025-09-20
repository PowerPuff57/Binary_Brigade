import sqlite3
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manage SQLite database for storing resumes, job descriptions and evaluations"""

    def __init__(self, db_path: str = "resume_evaluation.db"):
        self.db_path = db_path
        self._init_database()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_database(self):
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
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
        """)

        cursor.execute("""
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
        """)

        cursor.execute("""
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
            FOREIGN KEY (resume_id) REFERENCES resumes(id),
            FOREIGN KEY (job_id) REFERENCES job_descriptions(id)
        )
        """)

        conn.commit()
        conn.close()

    def save_resume(self, resume):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO resumes
            (id, name, email, phone, skills, experience, education, projects, certifications, summary, file_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
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
            resume.created_at.isoformat() if hasattr(resume.created_at, "isoformat") else str(resume.created_at)
        ))
        conn.commit()
        conn.close()

    def save_job_description(self, job_desc):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO job_descriptions
            (id, title, company, must_have_skills, good_to_have_skills, experience_required, education, description, location, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job_desc.id,
            job_desc.title,
            job_desc.company,
            json.dumps(job_desc.must_have_skills),
            json.dumps(job_desc.good_to_have_skills),
            job_desc.experience_required,
            json.dumps(job_desc.education),
            job_desc.description,
            job_desc.location,
            job_desc.created_at.isoformat() if hasattr(job_desc.created_at, "isoformat") else str(job_desc.created_at)
        ))
        conn.commit()
        conn.close()

    def save_evaluation(self, evaluation):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO evaluations
            (resume_id, job_id, relevance_score, hard_match_score, semantic_match_score, missing_skills, matched_skills, verdict, feedback, suggestions, evaluated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
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
            evaluation.evaluated_at.isoformat() if hasattr(evaluation.evaluated_at, "isoformat") else str(evaluation.evaluated_at)
        ))
        conn.commit()
        conn.close()

    def get_evaluations_for_job(self, job_id: str) -> List[Dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT e.id, e.resume_id, e.job_id, e.relevance_score, e.hard_match_score, e.semantic_match_score,
                   e.missing_skills, e.matched_skills, e.verdict, e.feedback, e.suggestions, e.evaluated_at,
                   r.name, r.email
            FROM evaluations e
            JOIN resumes r ON e.resume_id = r.id
            WHERE e.job_id = ?
            ORDER BY e.relevance_score DESC
        """, (job_id,))
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        results = []
        for row in rows:
            record = dict(zip(cols, row))
            try:
                record["missing_skills"] = json.loads(record.get("missing_skills") or "[]")
            except Exception:
                record["missing_skills"] = []
            try:
                record["matched_skills"] = json.loads(record.get("matched_skills") or "[]")
            except Exception:
                record["matched_skills"] = []
            try:
                record["suggestions"] = json.loads(record.get("suggestions") or "[]")
            except Exception:
                record["suggestions"] = []
            results.append(record)
        conn.close()
        return results

    def get_all_jobs(self) -> List[Dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM job_descriptions ORDER BY created_at DESC")
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        jobs = []
        for row in rows:
            record = dict(zip(cols, row))
            try:
                record["must_have_skills"] = json.loads(record.get("must_have_skills") or "[]")
            except Exception:
                record["must_have_skills"] = []
            try:
                record["good_to_have_skills"] = json.loads(record.get("good_to_have_skills") or "[]")
            except Exception:
                record["good_to_have_skills"] = []
            try:
                record["education"] = json.loads(record.get("education") or "[]")
            except Exception:
                record["education"] = []
            jobs.append(record)
        conn.close()
        return jobs
