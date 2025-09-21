# Resume Relevance Check System

An AI-powered resume evaluation system that automatically assesses candidate resumes against job descriptions and provides detailed relevance scoring, skill matching, and improvement suggestions.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Approach](#approach)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Problem Statement

HR teams and recruitment professionals face several challenges when evaluating resumes:

1. **Time-Intensive Manual Review**: Manually reviewing hundreds of resumes for each job posting is extremely time-consuming
2. **Inconsistent Evaluation Criteria**: Different reviewers may focus on different aspects, leading to inconsistent candidate assessment
3. **Skill Matching Complexity**: Identifying relevant technical skills and experience alignment requires domain expertise
4. **Scalability Issues**: As the number of applications increases, maintaining quality evaluation becomes difficult
5. **Bias in Resume Screening**: Human reviewers may unconsciously introduce bias based on factors unrelated to job requirements
6. **Missing Qualified Candidates**: Qualified candidates might be overlooked due to formatting differences or keyword mismatches

This system addresses these challenges by providing automated, consistent, and objective resume evaluation against job descriptions.

## Approach

### Core Methodology

The system employs a hybrid approach combining rule-based matching with natural language processing:

#### 1. Document Processing Pipeline
- **Multi-format Support**: Handles PDF, DOCX, and TXT files
- **Robust Text Extraction**: Uses multiple libraries (pdfplumber, PyPDF2, docx2txt) with fallback mechanisms
- **Text Normalization**: Cleans and standardizes extracted text for consistent processing

#### 2. Intelligent Parsing
- **Resume Parsing**: Extracts structured information including:
  - Personal details (name, email, phone)
  - Skills and technologies
  - Work experience and job history
  - Educational background
  - Projects and certifications
- **Job Description Parsing**: Identifies:
  - Required vs. preferred skills
  - Experience requirements
  - Educational qualifications
  - Job title and company information

#### 3. Multi-Dimensional Evaluation

##### Hard Match Scoring (60% weight)
- **Skill Alignment**: Direct keyword matching between resume skills and job requirements
- **Experience Matching**: Compares candidate experience level with job requirements
- **Educational Qualification**: Matches degree requirements with candidate education
- **Project Relevance**: Evaluates project experience against job domain
- **Certification Alignment**: Identifies relevant professional certifications

##### Semantic Match Scoring (40% weight - Optional)
- **Contextual Understanding**: Uses embeddings to understand semantic similarity
- **Domain Knowledge**: Identifies related skills and technologies
- **Experience Context**: Understands job role context beyond keyword matching

#### 4. Comprehensive Evaluation Output
- **Relevance Score**: Overall percentage score (0-100%)
- **Verdict Classification**: HIGH (75%+), MEDIUM (50-74%), LOW (<50%)
- **Detailed Feedback**: Specific areas of strength and improvement
- **Actionable Suggestions**: Concrete steps for candidate improvement

### Technical Implementation

#### Natural Language Processing
- **spaCy**: Named entity recognition for extracting personal information
- **NLTK**: Text tokenization and processing
- **Regular Expressions**: Pattern matching for emails, phones, skills
- **TF-IDF Vectorization**: Document similarity and keyword importance

#### Machine Learning Components
- **Scikit-learn**: Cosine similarity calculations
- **HuggingFace Embeddings**: Semantic text understanding (optional)
- **Fuzzy String Matching**: Handles variations in skill naming

#### Data Management
- **SQLite Database**: Stores job descriptions, resumes, and evaluation results
- **JSON Serialization**: Structured data storage and retrieval
- **File Management**: Secure file upload and storage

## Features

### Core Functionality
- **Multi-format Resume Processing**: PDF, DOCX, TXT support
- **Intelligent Job Description Parsing**: Automatic skill and requirement extraction
- **Comprehensive Evaluation**: Multi-factor scoring algorithm
- **Detailed Reporting**: In-depth analysis with actionable insights
- **Batch Processing**: Evaluate multiple resumes against single job description

### User Interface
- **Web Dashboard**: Clean, intuitive interface for HR teams
- **Real-time Processing**: Live feedback during upload and evaluation
- **Results Visualization**: Statistics and charts for evaluation results
- **Responsive Design**: Works on desktop and mobile devices

### Data Management
- **Persistent Storage**: All evaluations stored for future reference
- **Export Capabilities**: Results can be exported for reporting
- **Search and Filter**: Find specific evaluations quickly
- **Audit Trail**: Track all evaluations with timestamps

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   Database      │
│   (HTML/JS)     │    │   (Flask)       │    │   (SQLite)      │
│                 │    │                 │    │                 │
│ - Upload Forms  │◄──►│ - API Endpoints │◄──►│ - Job Desc.     │
│ - Results View  │    │ - File Process  │    │ - Resumes       │
│ - Statistics    │    │ - Evaluation    │    │ - Evaluations   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   NLP Pipeline  │
                       │                 │
                       │ - Text Extract  │
                       │ - Parsing       │
                       │ - Matching      │
                       │ - Scoring       │
                       └─────────────────┘
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- 4GB RAM minimum (8GB recommended)
- 1GB free disk space

### Step 1: Clone Repository

```bash
git clone https://github.com/your-username/resume-relevance-check.git
cd resume-relevance-check
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv resume_env

# Activate virtual environment
# On Windows:
resume_env\Scripts\activate
# On macOS/Linux:
source resume_env/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm
```

### Step 4: Create Requirements File

Create `requirements.txt` with the following content:

```txt
flask==2.3.3
flask-cors==4.0.0
PyPDF2==3.0.1
pdfplumber==0.9.0
docx2txt==0.8
python-docx==0.8.11
spacy==3.6.1
nltk==3.8.1
fuzzywuzzy==0.18.0
scikit-learn==1.3.0
numpy==1.24.3
werkzeug==2.3.7
python-Levenshtein==0.21.1
```

### Step 5: Verify Installation

```bash
# Run the application
python app.py

# You should see:
# ======================================
# Starting Resume Relevance Check System...
# Server will be available at: http://localhost:5000
# ======================================
```

### Optional: Advanced NLP Features

For enhanced semantic matching capabilities:

```bash
# Install additional ML libraries
pip install transformers torch sentence-transformers

# For OpenAI integration (requires API key)
pip install openai langchain
```

## Usage

### Starting the Application

1. **Start the Server**
   ```bash
   python app.py
   ```

2. **Access the Dashboard**
   Open your web browser and navigate to: `http://localhost:5000`

### Step-by-Step Workflow

#### 1. Upload Job Description

1. Click on **"Upload Job Description"** tab
2. Fill in company name and location
3. Either:
   - Paste job description text in the textarea, OR
   - Upload a PDF/DOCX file containing the job description
4. Click **"Upload Job Description"**
5. System will parse and extract:
   - Required skills
   - Preferred skills
   - Experience requirements
   - Educational qualifications

#### 2. Upload and Evaluate Resumes

1. Click on **"Upload Resume"** tab
2. Select the job description from dropdown
3. Upload resume file (PDF, DOCX, or TXT)
4. Click **"Evaluate Resume"**
5. System will:
   - Extract resume information
   - Compare against job requirements
   - Generate relevance score and feedback

#### 3. View Results

1. Click on **"View Results"** tab
2. Select job description to view all evaluations
3. Review:
   - **Statistics**: Total resumes, high/medium/low matches
   - **Individual Results**: Detailed evaluation for each candidate
   - **Skills Analysis**: Matched vs. missing skills
   - **Improvement Suggestions**: Actionable recommendations

### Understanding Results

#### Relevance Score
- **75-100%**: HIGH - Strong match, recommended for interview
- **50-74%**: MEDIUM - Partial match, consider with additional screening
- **0-49%**: LOW - Poor match, likely not suitable for role

#### Evaluation Components
- **Hard Match Score**: Direct skill and requirement matching
- **Semantic Match Score**: Contextual similarity (if enabled)
- **Missing Skills**: Critical skills the candidate lacks
- **Matched Skills**: Skills that align with job requirements
- **Suggestions**: Specific areas for candidate improvement

## API Documentation

### Base URL
```
http://localhost:5000/api
```

### Endpoints

#### Upload Job Description
```http
POST /api/upload_jd
Content-Type: multipart/form-data

Parameters:
- company (required): Company name
- location (optional): Job location
- jd_text (optional): Job description text
- jd_file (optional): Job description file

Response:
{
  "status": "success",
  "message": "Job description uploaded successfully!",
  "job_id": "abc123def456"
}
```

#### Upload Resume
```http
POST /api/upload_resume
Content-Type: multipart/form-data

Parameters:
- job_id (required): Job description ID
- resume_file (required): Resume file

Response:
{
  "status": "success",
  "message": "Resume evaluated successfully!",
  "evaluation": {
    "relevance_score": 78.5,
    "verdict": "HIGH",
    "matched_skills": ["python", "machine learning"],
    "missing_skills": ["docker", "kubernetes"],
    "suggestions": ["Add containerization experience"]
  }
}
```

#### Get All Jobs
```http
GET /api/jobs

Response:
[
  {
    "id": "abc123def456",
    "title": "Data Scientist",
    "company": "Tech Corp",
    "created_at": "2024-01-15T10:30:00"
  }
]
```

#### Get Evaluations for Job
```http
GET /api/evaluations/{job_id}

Response:
[
  {
    "name": "John Doe",
    "email": "john@example.com",
    "relevance_score": 78.5,
    "verdict": "HIGH",
    "feedback": "Strong technical background..."
  }
]
```

## Technical Details

### Supported File Formats

| Format | Extensions | Library Used | Notes |
|--------|------------|--------------|-------|
| PDF | .pdf | pdfplumber, PyPDF2 | Fallback mechanism for complex layouts |
| Word | .docx, .doc | docx2txt, python-docx | Modern and legacy Word formats |
| Text | .txt | Built-in | Plain text files |

### Database Schema

#### Job Descriptions Table
```sql
CREATE TABLE job_descriptions (
    id TEXT PRIMARY KEY,
    title TEXT,
    company TEXT,
    must_have_skills TEXT,  -- JSON array
    good_to_have_skills TEXT,  -- JSON array
    experience_required TEXT,
    education TEXT,  -- JSON array
    description TEXT,
    location TEXT,
    created_at TEXT
);
```

#### Resumes Table
```sql
CREATE TABLE resumes (
    id TEXT PRIMARY KEY,
    name TEXT,
    email TEXT,
    phone TEXT,
    skills TEXT,  -- JSON array
    experience TEXT,  -- JSON array
    education TEXT,  -- JSON array
    projects TEXT,  -- JSON array
    certifications TEXT,  -- JSON array
    summary TEXT,
    file_path TEXT,
    created_at TEXT
);
```

#### Evaluations Table
```sql
CREATE TABLE evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resume_id TEXT,
    job_id TEXT,
    relevance_score REAL,
    hard_match_score REAL,
    semantic_match_score REAL,
    missing_skills TEXT,  -- JSON array
    matched_skills TEXT,  -- JSON array
    verdict TEXT,
    feedback TEXT,
    suggestions TEXT,  -- JSON array
    evaluated_at TEXT
);
```

### Performance Considerations

- **File Size Limits**: 16MB maximum file size
- **Processing Time**: 2-5 seconds per resume evaluation
- **Concurrent Users**: Supports 10+ simultaneous users
- **Database**: SQLite suitable for 1000+ evaluations

## Troubleshooting

### Common Issues

#### 1. PDF Text Extraction Fails
**Symptoms**: Empty or garbled text from PDF files
**Solutions**:
- Ensure PDF contains selectable text (not scanned images)
- Try different PDF files to isolate the issue
- Check if pdfplumber and PyPDF2 are properly installed

#### 2. spaCy Model Not Found
**Symptoms**: `OSError: [E050] Can't find model 'en_core_web_sm'`
**Solution**:
```bash
python -m spacy download en_core_web_sm
```

#### 3. Port Already in Use
**Symptoms**: `Address already in use` error
**Solutions**:
- Change port in `app.py`: `app.run(debug=True, port=5001)`
- Kill existing process: `lsof -ti:5000 | xargs kill -9` (macOS/Linux)

#### 4. File Upload Fails
**Symptoms**: Upload button doesn't respond or files rejected
**Solutions**:
- Check file size (must be under 16MB)
- Verify file format (PDF, DOCX, TXT only)
- Clear browser cache and reload page

#### 5. No Skills Detected
**Symptoms**: Evaluation shows no matched skills
**Solutions**:
- Ensure resume contains clear skill sections
- Check if skills are written in standard format
- Verify job description contains specific skill requirements

### Debugging Mode

Enable detailed logging:

```python
# In app.py, change logging level
logging.basicConfig(level=logging.DEBUG)
```

### System Requirements Check

```bash
# Check Python version
python --version  # Should be 3.7+

# Check installed packages
pip list | grep -E "(flask|spacy|nltk|scikit-learn)"

# Test spaCy model
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('spaCy working!')"
```

## Contributing

We welcome contributions! Please follow these guidelines:

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Run tests: `python -m pytest tests/`
5. Submit pull request with detailed description

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings to functions and classes
- Include type hints where appropriate

### Testing

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_parsing.py

# Run with coverage
python -m pytest --cov=. tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions, issues, or feature requests:

1. **GitHub Issues**: Create an issue in the repository
2. **Documentation**: Check this README and code comments

## Acknowledgments

- **spaCy**: Advanced NLP library for text processing
- **Flask**: Web framework for the dashboard
- **scikit-learn**: Machine learning utilities
- **pdfplumber**: Robust PDF text extraction

---

**Version**: 1.0.0  
**Last Updated**: September 2025 
**Maintained by**: Binary Brigade
