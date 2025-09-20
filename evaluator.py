import json
import logging
import re
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Optional LangChain / embeddings / LLM imports (only used if enabled)
try:
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
except Exception:
    HuggingFaceEmbeddings = None
    ChatOpenAI = None
    PromptTemplate = None
    LLMChain = None

from models.resume import Resume
from models.job_description import JobDescription
from models.evaluation import EvaluationResult
from datetime import datetime

logger = logging.getLogger(__name__)


class RelevanceEvaluator:
    """Evaluate resume relevance against job descriptions"""

    def __init__(self, use_llm: bool = False, llm_api_key: Optional[str] = None):
        """
        :param use_llm: If True attempt to initialize embeddings + LLM (requires dependencies and API keys)
        :param llm_api_key: OpenAI API key (if using ChatOpenAI in langchain)
        """
        self.use_llm = use_llm and HuggingFaceEmbeddings is not None
        self.llm_api_key = llm_api_key

        self.embeddings = None
        self.llm = None

        if self.use_llm:
            try:
                self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                if llm_api_key and ChatOpenAI is not None:
                    self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=llm_api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize embeddings/LLM: {e}")
                self.embeddings = None
                self.llm = None

    def evaluate(self, resume: Resume, job_desc: JobDescription) -> EvaluationResult:
        """Main evaluation flow"""
        hard_result = self._hard_match_scoring(resume, job_desc)

        if self.embeddings:
            semantic_result = self._semantic_match_scoring(resume, job_desc)
        else:
            semantic_result = {"score": 0.0, "explanation": "Semantic evaluation disabled or embeddings not initialized."}

        hard_score = float(hard_result.get("score", 0.0))
        semantic_score = float(semantic_result.get("score", 0.0))

        final_score = (hard_score * 0.6) + (semantic_score * 0.4)
        verdict = self._determine_verdict(final_score)

        feedback = self._generate_feedback(resume, job_desc, hard_result, semantic_result, final_score)
        suggestions = self._generate_suggestions(resume, job_desc, hard_result.get("missing_skills", []))

        return EvaluationResult(
            resume_id=resume.id,
            job_id=job_desc.id,
            relevance_score=round(final_score, 2),
            hard_match_score=round(hard_score, 2),
            semantic_match_score=round(semantic_score, 2),
            missing_skills=hard_result.get("missing_skills", []),
            matched_skills=hard_result.get("matched_skills", []),
            verdict=verdict,
            feedback=feedback,
            suggestions=suggestions,
            evaluated_at=datetime.now()
        )

    def _hard_match_scoring(self, resume: Resume, job_desc: JobDescription) -> Dict:
        """Score based on explicit matches (skills, education, experience, projects, certs)"""
        score = 0.0
        matched_skills = []
        missing_skills = []

        resume_skills = set([s.lower() for s in (resume.skills or [])])
        must_have = set([s.lower() for s in (job_desc.must_have_skills or [])])
        good_to_have = set([s.lower() for s in (job_desc.good_to_have_skills or [])])

        # Must-have skills (up to 30)
        if must_have:
            matched_must = resume_skills.intersection(must_have)
            matched_skills.extend(matched_must)
            missing = must_have - matched_must
            missing_skills.extend(list(missing))
            score += (len(matched_must) / len(must_have)) * 30.0

        # Good-to-have (up to 10)
        if good_to_have:
            matched_good = resume_skills.intersection(good_to_have)
            matched_skills.extend(matched_good)
            score += (len(matched_good) / len(good_to_have)) * 10.0

        # Education (15)
        if job_desc.education:
            resume_edu_text = " ".join([edu.get("degree", "") for edu in (resume.education or [])]).lower()
            edu_matched = any(req.lower() in resume_edu_text for req in job_desc.education)
            if edu_matched:
                score += 15.0

        # Experience (15)
        if job_desc.experience_required and job_desc.experience_required != "Not specified":
            years_match = re.search(r"(\d+)", job_desc.experience_required)
            if years_match:
                required_years = int(years_match.group(1))
                # Rough estimate: assume each experience entry ~2 years (same heuristic used in original)
                candidate_years = max(0, len(resume.experience or []) * 2)
                if candidate_years >= required_years:
                    score += 15.0
                elif candidate_years >= required_years * 0.7:
                    score += 10.0
                elif candidate_years >= required_years * 0.5:
                    score += 5.0

        # Projects relevance (up to 15)
        if resume.projects:
            project_text = " ".join([(p.get("name", "") + " " + p.get("description", "")) for p in resume.projects]).lower()
            relevant_keywords = must_have.union(good_to_have)
            relevance_count = sum(1 for keyword in relevant_keywords if keyword in project_text)
            if relevance_count > 0:
                score += min(15.0, relevance_count * 3.0)

        # Certifications (up to 15)
        if resume.certifications:
            cert_text = " ".join(resume.certifications).lower()
            cert_count = sum(1 for skill in must_have.union(good_to_have) if skill in cert_text)
            if cert_count > 0:
                score += min(15.0, cert_count * 5.0)

        return {
            "score": min(100.0, score),
            "matched_skills": list(set(matched_skills)),
            "missing_skills": list(set(missing_skills))
        }

    def _semantic_match_scoring(self, resume: Resume, job_desc: JobDescription) -> Dict:
        """Use embeddings to compute semantic similarity (converted to 0-100 scale)"""
        if not self.embeddings:
            return {"score": 0.0, "explanation": "Embeddings not available."}
        try:
            # embed_query returns a vector for LangChain HuggingFaceEmbeddings
            resume_emb = self.embeddings.embed_query(resume.raw_text[:3000])
            jd_emb = self.embeddings.embed_query(job_desc.description[:3000])
            sim = cosine_similarity(
                np.array(resume_emb).reshape(1, -1),
                np.array(jd_emb).reshape(1, -1)
            )[0][0]
            score = float(sim) * 100.0

            explanation = f"Semantic similarity: {sim:.3f}"
            # Optional: generate LLM explanation if llm available
            if self.llm and PromptTemplate and LLMChain:
                try:
                    prompt = PromptTemplate(
                        template="""
Analyze the semantic match between this resume and job description in 2-3 sentences.

Resume summary: {resume_summary}
Key resume skills: {resume_skills}

Job title: {job_title}
Required skills: {required_skills}

Similarity: {similarity}
""",
                        input_variables=["resume_summary", "resume_skills", "job_title", "required_skills", "similarity"]
                    )
                    chain = LLMChain(llm=self.llm, prompt=prompt)
                    explanation_text = chain.run(
                        resume_summary=resume.summary[:200] if resume.summary else "Not provided",
                        resume_skills=", ".join(resume.skills[:10]) if resume.skills else "None",
                        job_title=job_desc.title,
                        required_skills=", ".join(job_desc.must_have_skills[:10]) if job_desc.must_have_skills else "None",
                        similarity=f"{sim:.3f}"
                    )
                    explanation = explanation_text.strip()
                except Exception as e:
                    logger.debug(f"LLM explanation generation error: {e}")

            return {"score": score, "explanation": explanation}
        except Exception as e:
            logger.error(f"Semantic matching failed: {e}")
            return {"score": 0.0, "explanation": "Semantic matching error."}

    def _determine_verdict(self, score: float) -> str:
        if score >= 75:
            return "HIGH"
        elif score >= 50:
            return "MEDIUM"
        else:
            return "LOW"

    def _generate_feedback(self, resume: Resume, job_desc: JobDescription, hard_match: Dict, semantic_match: Dict, final_score: float) -> str:
        parts = []
        parts.append(f"Overall Relevance Score: {final_score:.1f}/100")
        if hard_match.get("matched_skills"):
            parts.append(f"Matched Skills: {', '.join(hard_match['matched_skills'][:5])}")
        if hard_match.get("missing_skills"):
            parts.append(f"Missing Key Skills: {', '.join(hard_match['missing_skills'][:5])}")
        if job_desc.experience_required and job_desc.experience_required != "Not specified":
            est_years = max(0, len(resume.experience or []) * 2)
            parts.append(f"Estimated Experience: ~{est_years} years (Required: {job_desc.experience_required})")
        if resume.projects:
            parts.append(f"Projects Found: {len(resume.projects)}")
        else:
            parts.append("No projects mentioned - consider adding relevant projects.")
        if semantic_match.get("explanation"):
            parts.append(f"Content Analysis: {semantic_match.get('explanation')}")
        return " | ".join(parts)

    def _generate_suggestions(self, resume: Resume, job_desc: JobDescription, missing_skills: List[str]) -> List[str]:
        suggestions = []
        if missing_skills:
            suggestions.append(f"Acquire these critical skills: {', '.join(missing_skills[:3])}")
        if not resume.projects or len(resume.projects) < 2:
            suggestions.append("Add 2-3 relevant projects demonstrating practical experience.")
        if not resume.certifications:
            suggestions.append("Consider obtaining relevant certifications in your domain.")
        if len(resume.experience or []) < 2:
            suggestions.append("Gain more practical experience through internships or freelance work.")
        if not resume.summary or len(resume.summary or "") < 50:
            suggestions.append("Add a compelling professional summary highlighting your key strengths.")
        return suggestions[:5]
