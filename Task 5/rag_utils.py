"""
rag_utils.py
------------
Shared logic for the "RAG-Powered Talent Search Engine" (Task 8).

This module is imported by both:
  - Task8_RAG_Talent_Search.ipynb  (build / experiment / evaluate)
  - streamlit_app.py               (chat UI for recruiters)

It implements:
  1. Loading & parsing the Kaggle "Resume Entities for NER" dataset
     (with a synthetic fallback so the pipeline always runs end-to-end).
  2. Embedding resumes with a HuggingFace sentence-transformer.
  3. Storing/retrieving them from a Chroma vector database.
  4. Semantic search over resumes given a natural-language recruiter query.
  5. LLM-based evaluation: turning the top-k retrieved resumes into a
     human-readable "why this candidate fits" summary (OpenAI if a key is
     available, else a local HF model, else a transparent rule-based
     fallback -- so the notebook never hard-fails just because no API key
     is configured).
  6. A basic "Bias Check" that looks at whether retrieval results are
     skewed by category, resume length, or (as an illustrative, clearly
     caveated proxy) gender-coded first names.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Callable

# ----------------------------------------------------------------------
# 0. Config
# ----------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = "chroma_db"
CHROMA_COLLECTION_NAME = "resumes"

# Common technical skill keywords we look for when the raw dataset does not
# already provide clean "Skills" entity spans (used both for the synthetic
# fallback data and for lightweight keyword extraction on real resumes).
SKILL_VOCAB = [
    "python", "java", "c++", "sql", "nosql", "excel", "tableau", "power bi",
    "machine learning", "deep learning", "nlp", "computer vision",
    "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
    "aws", "azure", "gcp", "docker", "kubernetes", "spark", "hadoop",
    "airflow", "etl", "data warehousing", "data modeling", "statistics",
    "a/b testing", "leadership", "project management", "agile", "scrum",
    "react", "node.js", "django", "flask", "html", "css", "javascript",
    "r", "sas", "spss", "git", "linux", "communication", "stakeholder management",
]


# ----------------------------------------------------------------------
# 1. Data loading
# ----------------------------------------------------------------------

@dataclass
class Resume:
    id: str
    name: str
    category: str
    text: str
    skills: List[str] = field(default_factory=list)
    experience_level: str = "Unknown"

    def to_metadata(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "skills": ", ".join(self.skills),
            "experience_level": self.experience_level,
            "length_chars": len(self.text),
        }


def _guess_experience_level(text: str) -> str:
    """Very rough heuristic used only for demo metadata / bias analysis."""
    t = text.lower()
    if any(k in t for k in ["senior", "lead", "8+ years", "10+ years", "principal", "manager"]):
        return "Senior"
    if any(k in t for k in ["junior", "entry level", "intern", "fresher", "0-1 year", "1 year"]):
        return "Junior"
    return "Mid"


def _extract_skills(text: str) -> List[str]:
    t = text.lower()
    return sorted({skill for skill in SKILL_VOCAB if skill in t})


def load_resumes_from_dataturks_json(path: str) -> List[Resume]:
    """
    Parses the Kaggle "Resume Entities for NER" dataset, which is shipped
    in DataTurks JSON-lines annotation format. Each line looks like:

    {
      "content": "<full resume text>",
      "annotation": [
          {"label": ["Skills"], "points": [{"text": "Python", ...}]},
          {"label": ["Name"], "points": [{"text": "John Doe", ...}]},
          ...
      ]
    }

    We pull out the raw text plus any "Skills" / "Designation" / "Name"
    entity spans that are present, and fall back to keyword extraction
    when a resume has no explicit "Skills" annotations.
    """
    _SPLIT_RE = re.compile(r"[•\n,;|]+")

    def _tokenize_skill_span(span: str) -> List[str]:
        """The raw 'Skills' annotation spans in this dataset range from a
        clean comma list to a whole noisy paragraph. Split on common
        delimiters and keep short, plausible skill tokens."""
        tokens = []
        for raw_tok in _SPLIT_RE.split(span):
            tok = raw_tok.strip(" \t-•.")
            tok = re.sub(r"\(.*?\)", "", tok).strip()  # drop "(Less than 1 year)" etc.
            if 1 < len(tok) <= 40 and not tok.lower().startswith(("technical skill", "non - technical", "additional information")):
                tokens.append(tok.lower())
        return tokens

    resumes: List[Resume] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = record.get("content", "").strip()
            if not text:
                continue

            skill_tokens: List[str] = []
            name = None
            designation_candidates: List[str] = []
            for ann in record.get("annotation", []):
                labels = ann.get("label", [])
                points = ann.get("points", [])
                if not points:
                    continue
                span_text = points[0].get("text", "").strip()
                if not span_text:
                    continue
                if "Skills" in labels:
                    skill_tokens.extend(_tokenize_skill_span(span_text))
                elif "Name" in labels and name is None:
                    name = span_text.split("\n")[0].strip()
                elif "Designation" in labels:
                    designation_candidates.append(span_text)

            # Pick the cleanest (single-line, reasonably short) designation
            # as the "category"; the raw annotations are noisy and sometimes
            # bundle multiple lines/entities together.
            category = "General"
            for cand in designation_candidates:
                first_line = cand.split("\n")[0].strip()
                if first_line and len(first_line) <= 60:
                    category = first_line
                    break

            skills = sorted(set(skill_tokens)) if skill_tokens else _extract_skills(text)

            resumes.append(
                Resume(
                    id=f"resume_{i}",
                    name=name or f"Candidate {i+1}",
                    category=category,
                    text=text,
                    skills=skills,
                    experience_level=_guess_experience_level(text),
                )
            )
    return resumes


def load_synthetic_resumes() -> List[Resume]:
    """
    A small, hand-written fallback dataset so the whole pipeline (embedding,
    retrieval, LLM evaluation, bias check, Streamlit chat) can be run and
    demoed even before the real Kaggle dataset has been downloaded.
    Replace this with `load_resumes_from_dataturks_json(path)` once you
    have the real file (see README cell in the notebook for the download link).
    """
    raw = [
        dict(
            name="Aisha Rahman",
            category="Data Analyst",
            experience_level="Junior",
            text=(
                "Aisha Rahman - Junior Data Analyst. 1 year of experience turning raw "
                "data into actionable insight. Strong in SQL, Excel and Tableau for "
                "dashboarding. Built weekly sales dashboards used by regional managers. "
                "Comfortable with Python (pandas) for data cleaning and basic statistics. "
                "BSc in Statistics."
            ),
        ),
        dict(
            name="David Chen",
            category="Data Scientist",
            experience_level="Senior",
            text=(
                "David Chen - Senior Data Scientist with 9+ years of experience leading "
                "machine learning initiatives. Expert in Python, scikit-learn, TensorFlow "
                "and PyTorch. Led a team of 4 data scientists building churn-prediction "
                "models. Strong stakeholder management and leadership experience across "
                "cross-functional teams. Also fluent in SQL and AWS."
            ),
        ),
        dict(
            name="Maria Lopez",
            category="Data Analyst",
            experience_level="Mid",
            text=(
                "Maria Lopez - Data Analyst, 3 years experience. Specializes in SQL, "
                "Tableau and Power BI reporting for marketing teams. Ran multiple A/B "
                "tests to measure campaign performance. Some Python scripting for ETL. "
                "Known for clear communication of insights to non-technical stakeholders."
            ),
        ),
        dict(
            name="Samuel Okafor",
            category="Software Engineer",
            experience_level="Junior",
            text=(
                "Samuel Okafor - Junior Software Engineer, entry level, 6 months "
                "experience. Skilled in Java, React and Node.js. Built a small internal "
                "tool using Flask and PostgreSQL. Familiar with Git and Agile/Scrum "
                "ceremonies. Eager to grow into backend engineering."
            ),
        ),
        dict(
            name="Priya Sharma",
            category="Data Analyst",
            experience_level="Junior",
            text=(
                "Priya Sharma - Junior Data Analyst with 8 months of internship "
                "experience. Proficient in SQL and Tableau, learning Power BI. Assisted "
                "senior analysts with data cleaning in Excel and basic Python scripts. "
                "Strong academic background in statistics and eager to take ownership of "
                "small reporting projects."
            ),
        ),
        dict(
            name="James Whitfield",
            category="Project Manager",
            experience_level="Senior",
            text=(
                "James Whitfield - Senior Project Manager, 12 years experience managing "
                "cross-functional engineering and data teams. Strong leadership, "
                "stakeholder management and Agile/Scrum expertise. Some exposure to SQL "
                "for reporting but not hands-on with data science tools."
            ),
        ),
        dict(
            name="Lena Fischer",
            category="Machine Learning Engineer",
            experience_level="Mid",
            text=(
                "Lena Fischer - Machine Learning Engineer, 4 years experience deploying "
                "models with Docker and Kubernetes on AWS. Strong Python, PyTorch and "
                "NLP background. Built a computer vision pipeline for defect detection. "
                "Comfortable with SQL for feature engineering."
            ),
        ),
        dict(
            name="Omar Haddad",
            category="Data Analyst",
            experience_level="Junior",
            text=(
                "Omar Haddad - Junior Data Analyst, 1.5 years experience at a retail "
                "analytics startup. Core skills: SQL, Tableau, Excel, basic Python for "
                "automating reports. Presented monthly KPI dashboards to leadership. "
                "Studying for AWS Cloud Practitioner certification."
            ),
        ),
        dict(
            name="Grace Kim",
            category="HR Specialist",
            experience_level="Mid",
            text=(
                "Grace Kim - HR Specialist, 5 years experience in recruiting and people "
                "operations. Strong communication and project management skills. Uses "
                "Excel for headcount reporting; no programming or SQL background."
            ),
        ),
        dict(
            name="Yusuf Demir",
            category="Data Analyst",
            experience_level="Mid",
            text=(
                "Yusuf Demir - Data Analyst, 3.5 years experience in the finance sector. "
                "Advanced SQL, Tableau and Excel skills. Automates recurring reports with "
                "Python. Has mentored two junior analysts and led a small reporting "
                "workstream (light leadership experience)."
            ),
        ),
        dict(
            name="Fatima Al-Sayed",
            category="Data Scientist",
            experience_level="Junior",
            text=(
                "Fatima Al-Sayed - Junior Data Scientist, entry level, 1 year experience "
                "including internship. Skilled in Python, pandas, scikit-learn and SQL. "
                "Built a customer segmentation model as a capstone project. Learning "
                "Tableau for visualization."
            ),
        ),
        dict(
            name="Tom Becker",
            category="Software Engineer",
            experience_level="Senior",
            text=(
                "Tom Becker - Senior Software Engineer, 10 years experience, strong in "
                "Java, C++ and distributed systems (Spark, Hadoop). Led architecture "
                "decisions and mentored junior engineers (leadership experience). Limited "
                "exposure to Tableau or dashboarding tools."
            ),
        ),
    ]
    resumes = []
    for i, r in enumerate(raw):
        resumes.append(
            Resume(
                id=f"resume_{i}",
                name=r["name"],
                category=r["category"],
                text=r["text"],
                skills=_extract_skills(r["text"]),
                experience_level=r["experience_level"],
            )
        )
    return resumes


def load_resumes(dataset_path: Optional[str] = None) -> List[Resume]:
    """
    Tries, in order:
      1. `dataset_path` if given and it exists (real Kaggle DataTurks JSON).
      2. A handful of common filename variants for this dataset (the Kaggle
         download is named "Entity Recognition in Resumes.json", but
         uploaders / zip tools frequently rewrite the spaces to underscores
         or hyphens, e.g. "Entity_Recognition_in_Resumes.json") in both the
         current directory and a "data/" subfolder.
      3. Any *.json file in the current directory / "data/" whose name
         contains "resum" or "entity" (case-insensitive) -- a last-resort
         sweep so a renamed file is still picked up automatically.
      4. Falls back to the built-in synthetic dataset so the notebook always
         runs, and says so loudly (this should never happen silently).
    """
    _NAME_VARIANTS = [
        "Entity Recognition in Resumes.json",
        "Entity_Recognition_in_Resumes.json",
        "Entity-Recognition-in-Resumes.json",
        "entity_recognition_in_resumes.json",
    ]

    candidates = []
    if dataset_path:
        candidates.append(dataset_path)
    for name in _NAME_VARIANTS:
        candidates.append(name)
        candidates.append(f"data/{name}")

    # Last-resort sweep: any json file that looks like this dataset, in cwd
    # or ./data, so a differently-renamed copy is still found.
    for folder in (".", "data"):
        p = Path(folder)
        if p.is_dir():
            for f in p.glob("*.json"):
                lname = f.name.lower()
                if "resum" in lname or "entity" in lname:
                    candidates.append(str(f))

    tried = []
    for path in candidates:
        if not path or path in tried:
            continue
        tried.append(path)
        if Path(path).exists():
            try:
                resumes = load_resumes_from_dataturks_json(path)
                if resumes:
                    print(f"Loaded {len(resumes)} resumes from '{path}'.")
                    return resumes
            except Exception as e:
                print(f"Could not parse '{path}' ({e}); trying next source.")

    print(
        "WARNING: real dataset not found on disk (looked for: "
        f"{', '.join(tried)}) -> falling back to the built-in synthetic demo "
        "dataset (12 sample resumes). Pass the correct path explicitly via "
        "load_resumes(dataset_path=...) to use real data."
    )
    return load_synthetic_resumes()


# ----------------------------------------------------------------------
# 2. Vector store (embeddings + Chroma)
# ----------------------------------------------------------------------

def build_vector_store(resumes: List[Resume], persist_dir: str = CHROMA_PERSIST_DIR):
    """
    Embeds resumes with a HuggingFace sentence-transformer and stores them
    in a persistent Chroma collection. Returns the Chroma vectorstore object.
    """
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings

    # The "Document" class has moved between packages across langchain
    # versions. Try every known location so this works regardless of which
    # exact langchain/langchain-core/langchain-community versions ended up
    # installed.
    Document = None
    for import_path in (
        "langchain_core.documents",
        "langchain.docstore.document",
        "langchain.schema",
        "langchain_community.docstore.document",
    ):
        try:
            _mod = __import__(import_path, fromlist=["Document"])
            Document = _mod.Document
            break
        except (ImportError, AttributeError):
            continue
    if Document is None:
        raise ImportError(
            "Could not locate the 'Document' class in any known langchain "
            "location. Try upgrading: pip install -U langchain langchain-core "
            "langchain-community"
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    docs = [
        Document(page_content=r.text, metadata=r.to_metadata())
        for r in resumes
    ]

    # Re-running this cell (e.g. after a reload/restart) previously *appended*
    # to whatever was already on disk at persist_dir, silently doubling the
    # collection. Explicitly drop any pre-existing collection with the same
    # name first so this function is safe to call more than once.
    try:
        _tmp_client = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        _tmp_client.delete_collection()
    except Exception:
        pass  # nothing to delete yet -- fine on a first run

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    return vectorstore


def load_vector_store(persist_dir: str = CHROMA_PERSIST_DIR):
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


def semantic_search(vectorstore, query: str, k: int = 3) -> List[Dict]:
    """
    Runs similarity search and returns a clean list of
    {name, category, skills, experience_level, text, score} dicts,
    sorted best-first (lower distance / higher similarity first).
    """
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    out = []
    for doc, score in results:
        meta = doc.metadata
        out.append(
            {
                "id": meta.get("id"),
                "name": meta.get("name"),
                "category": meta.get("category"),
                "skills": meta.get("skills", ""),
                "experience_level": meta.get("experience_level"),
                "length_chars": meta.get("length_chars"),
                "text": doc.page_content,
                "score": round(float(score), 4),
            }
        )
    return out


# ----------------------------------------------------------------------
# 3. LLM-based evaluation ("why does this candidate fit?")
# ----------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "will", "with", "who", "who's", "knows", "know", "find",
    "me", "you", "your",
}


def _rule_based_summary(query: str, candidate: Dict) -> str:
    """
    Zero-dependency fallback used when no OpenAI key and no local HF
    generation model is available. Not a real LLM call, but keeps the
    pipeline usable offline / without API keys. Still highlights concrete
    overlap between the query and the resume, which is the point of the
    evaluation step.
    """
    query_terms = set(re.findall(r"[a-zA-Z\+\#\.]+", query.lower())) - _STOPWORDS
    resume_terms = set(re.findall(r"[a-zA-Z\+\#\.]+", candidate["text"].lower())) - _STOPWORDS
    overlap = sorted(t for t in query_terms & resume_terms if len(t) > 2)
    skills = candidate.get("skills", "")
    return (
        f"{candidate['name']} ({candidate['experience_level']} {candidate['category']}) "
        f"is a plausible match for the query \"{query}\". "
        f"Matched keywords: {', '.join(overlap) if overlap else 'general profile similarity'}. "
        f"Listed skills: {skills or 'not explicitly extracted'}. "
        f"[Generated by rule-based fallback -- configure OPENAI_API_KEY or a local "
        f"HF model for a richer LLM-written explanation.]"
    )


_PROMPT_FIELD_LABELS = (
    "category:", "experience level:", "extracted skills:", "resume excerpt:", "name:",
)


def _looks_degenerate(text: str) -> bool:
    """
    Heuristic guard against the small local flan-t5-base model's two failure
    modes seen in practice:
      1. Echoing the prompt's structured field labels verbatim instead of
         writing a real sentence (e.g. "Category: Systems Engineer
         Experience level: Senior Extracted skills: ...").
      2. Collapsing to something too short/empty to be useful.
    Returns True when the output should be discarded in favor of the
    rule-based fallback rather than shown to a recruiter as-is.
    """
    if not text:
        return True
    words = text.split()
    if len(words) < 4:
        return True
    lower = text.lower()
    label_hits = sum(1 for label in _PROMPT_FIELD_LABELS if label in lower)
    # Two or more raw field labels showing up verbatim is a strong signal
    # the model just copied the prompt's structured header instead of
    # generating an explanation.
    if label_hits >= 2:
        return True
    return False


def get_llm() -> Callable[[str], str]:
    """
    Returns a single callable `llm(prompt) -> str`.
    Tries, in priority order:
      1. OpenAI chat completion (if OPENAI_API_KEY is set).
      2. A small local HuggingFace text2text model (flan-t5-base).
      3. The rule-based fallback (always available, no dependencies).
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            def _call(prompt: str) -> str:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.3,
                )
                return resp.choices[0].message.content.strip()

            print("Using OpenAI (gpt-4o-mini) for LLM-based evaluation.")
            return _call
        except Exception as e:
            print(f"OpenAI unavailable ({e}); trying local HF model.")

    try:
        # Load the seq2seq model directly instead of going through
        # transformers.pipeline("text2text-generation", ...). Some recent
        # transformers/huggingface_hub combinations route pipeline() task
        # names through a different (Inference API) task registry that
        # doesn't recognize "text2text-generation", raising a confusing
        # "Unknown task" error even though the library is installed fine.
        # Loading the model/tokenizer directly sidesteps that entirely.
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        _model_name = "google/flan-t5-base"
        _tokenizer = AutoTokenizer.from_pretrained(_model_name)
        _model = AutoModelForSeq2SeqLM.from_pretrained(_model_name)

        def _call(prompt: str) -> str:
            inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            output_ids = _model.generate(
                **inputs,
                max_new_tokens=150,
                min_new_tokens=25,
                do_sample=False,
                num_beams=4,
                # flan-t5-base tends to degenerate into copying chunks of a
                # long resume verbatim (or collapsing to a single word) on
                # greedy/short-input decoding. Beam search + a repetition
                # penalty + a no-repeat n-gram constraint keeps it producing
                # an actual short explanation instead.
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                early_stopping=True,
            )
            text = _tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            if _looks_degenerate(text):
                return None
            return text

        print("Using local HuggingFace model (flan-t5-base) for LLM-based evaluation.")
        return _call
    except Exception as e:
        print(f"Local HF model unavailable ({e}); using rule-based fallback for evaluation.")

    def _fallback(prompt: str) -> str:
        return None  # signals caller to use _rule_based_summary directly

    return _fallback


def build_fit_prompt(query: str, candidate: Dict, max_resume_chars: int = 900) -> str:
    # Long, noisy resume text is what causes small local models to just copy
    # a chunk of it back verbatim instead of reasoning about fit. Keeping
    # the prompt short and leading with the structured fields (skills,
    # category, level) -- which is what actually drives the "why this
    # candidate fits" judgement -- and trimming the raw text keeps every
    # backend (OpenAI or local) focused on producing an explanation rather
    # than a transcript.
    resume_excerpt = candidate["text"][:max_resume_chars]
    if len(candidate["text"]) > max_resume_chars:
        resume_excerpt += " [...]"
    return (
        "You are a recruiting assistant. A recruiter is searching for candidates "
        f"matching this requirement: \"{query}\".\n\n"
        "Here is one retrieved candidate:\n"
        f"Name: {candidate['name']}\n"
        f"Category: {candidate['category']}\n"
        f"Experience level: {candidate['experience_level']}\n"
        f"Extracted skills: {candidate['skills']}\n"
        f"Resume excerpt: {resume_excerpt}\n\n"
        "Write 2-3 short sentences, in your own words (do not copy resume "
        "sentences verbatim), explaining specifically why this candidate is "
        "(or is not) a good fit for the requirement, referencing concrete "
        "skills or experience. Be honest about any gaps."
    )


def evaluate_top_candidates(query: str, candidates: List[Dict], llm: Optional[Callable] = None) -> List[Dict]:
    """
    For each candidate, generates an LLM-written "why this candidate fits"
    explanation and attaches it as candidate["llm_summary"].
    """
    if llm is None:
        llm = get_llm()

    enriched = []
    for c in candidates:
        prompt = build_fit_prompt(query, c)
        summary = None
        try:
            summary = llm(prompt)
        except Exception as e:
            summary = None
        if not summary:
            summary = _rule_based_summary(query, c)
        c = dict(c)
        c["llm_summary"] = summary
        enriched.append(c)
    return enriched


# ----------------------------------------------------------------------
# 4. Bias check
# ----------------------------------------------------------------------

# NOTE: this is an illustrative, heavily-caveated heuristic. The dataset has
# no real demographic labels, so we approximate with a small first-name list.
# This is NOT reliable for real hiring decisions -- it is only meant to
# demonstrate *how* you would wire up a bias-monitoring check in a real
# pipeline (where you would use validated, consented demographic data
# instead of inferring it from names).
_ILLUSTRATIVE_NAME_HINTS = {
    "aisha": "F", "maria": "F", "priya": "F", "lena": "F", "grace": "F", "fatima": "F",
    "david": "M", "samuel": "M", "james": "M", "omar": "M", "yusuf": "M", "tom": "M",
}


def _name_hint(name: str) -> str:
    first = name.strip().split()[0].lower() if name else ""
    return _ILLUSTRATIVE_NAME_HINTS.get(first, "Unknown")


def bias_check(retrieved: List[Dict], corpus: List[Resume]) -> Dict:
    """
    Compares the retrieved set against the overall corpus along three axes:
      1. Category distribution skew (e.g. always returning "Data Scientist"
         even when other categories are equally relevant).
      2. Resume length bias (does retrieval systematically favor longer resumes?).
      3. Illustrative gender-name-hint distribution (heavily caveated).

    Returns a dict with counts + simple textual flags. This is a starting
    point for a real fairness audit, not a certified bias metric.
    """
    report: Dict = {"flags": []}

    # 1. Category skew
    corpus_categories = [r.category for r in corpus]
    retrieved_categories = [c["category"] for c in retrieved]
    corpus_cat_dist = {c: corpus_categories.count(c) / len(corpus_categories) for c in set(corpus_categories)}
    retrieved_cat_dist = {c: retrieved_categories.count(c) / len(retrieved_categories) for c in set(retrieved_categories)}
    report["corpus_category_distribution"] = corpus_cat_dist
    report["retrieved_category_distribution"] = retrieved_cat_dist

    for cat, ret_share in retrieved_cat_dist.items():
        base_share = corpus_cat_dist.get(cat, 0)
        if base_share > 0 and ret_share > base_share * 2 and ret_share > 0.5:
            report["flags"].append(
                f"Category '{cat}' is over-represented in results ({ret_share:.0%} of "
                f"retrieved vs {base_share:.0%} of the corpus)."
            )

    # 2. Length bias
    avg_len_corpus = sum(len(r.text) for r in corpus) / len(corpus)
    avg_len_retrieved = sum(c["length_chars"] for c in retrieved) / len(retrieved)
    report["avg_resume_length_corpus"] = round(avg_len_corpus, 1)
    report["avg_resume_length_retrieved"] = round(avg_len_retrieved, 1)
    if avg_len_retrieved > avg_len_corpus * 1.5:
        report["flags"].append(
            "Retrieved resumes are, on average, notably longer than the corpus "
            "average -- possible bias toward verbose resumes rather than relevance."
        )

    # 3. Illustrative name-hint distribution
    corpus_hints = [_name_hint(r.name) for r in corpus]
    retrieved_hints = [_name_hint(c["name"]) for c in retrieved]
    corpus_hint_dist = {h: corpus_hints.count(h) / len(corpus_hints) for h in set(corpus_hints)}
    retrieved_hint_dist = {h: retrieved_hints.count(h) / len(retrieved_hints) for h in set(retrieved_hints)}
    report["illustrative_name_hint_distribution_corpus"] = corpus_hint_dist
    report["illustrative_name_hint_distribution_retrieved"] = retrieved_hint_dist
    report["name_hint_disclaimer"] = (
        "Gender hints are inferred from a tiny hard-coded first-name list for "
        "demonstration only. They are unreliable and should never be used for "
        "real hiring decisions -- use consented, validated demographic data "
        "and a proper fairness library (e.g. Fairlearn, AIF360) in production."
    )

    if not report["flags"]:
        report["flags"].append("No strong skew detected on the axes checked above (small sample size -- treat as illustrative).")

    return report


def format_bias_report(report: Dict) -> str:
    lines = ["## Bias Check Report", ""]

    corpus_dist = report["corpus_category_distribution"]
    retrieved_dist = report["retrieved_category_distribution"]

    # This dataset's "category" comes from noisy, un-normalized raw job-title
    # text (e.g. "Senior System Engineer at Infosys Limited" as its own
    # distinct category), so a real corpus can have 100+ unique values.
    # Printing every corpus category (almost all at 0% retrieved) makes the
    # report unreadable. Show only the categories that actually appear in
    # the retrieved set -- which is exactly what a recruiter needs to judge
    # skew -- plus a note on how many other categories exist in the corpus
    # for context.
    lines.append("**Category distribution (corpus vs retrieved), for categories present in the results:**")
    shown_cats = sorted(retrieved_dist, key=lambda c: -retrieved_dist[c])
    for cat in shown_cats:
        r = retrieved_dist[cat]
        c = corpus_dist.get(cat, 0)
        lines.append(f"- {cat}: corpus {c:.0%} -> retrieved {r:.0%}")
    other_cats = len(corpus_dist) - len(set(corpus_dist) & set(retrieved_dist))
    if other_cats > 0:
        lines.append(
            f"- _(+{other_cats} other categories in the corpus, 0% of this result set -- "
            "omitted here for readability)_"
        )
    lines.append("")
    lines.append(
        f"**Avg resume length:** corpus {report['avg_resume_length_corpus']} chars vs "
        f"retrieved {report['avg_resume_length_retrieved']} chars"
    )
    lines.append("")
    lines.append("**Illustrative gender-name-hint distribution:**")
    lines.append(f"- Corpus: {report['illustrative_name_hint_distribution_corpus']}")
    lines.append(f"- Retrieved: {report['illustrative_name_hint_distribution_retrieved']}")
    lines.append(f"- _{report['name_hint_disclaimer']}_")
    lines.append("")
    lines.append("**Flags:**")
    for f in report["flags"]:
        lines.append(f"- {f}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# 5. Chat-style Q&A over a single candidate (for the Streamlit UI)
# ----------------------------------------------------------------------

def answer_question_about_candidate(question: str, candidate: Dict, llm: Optional[Callable] = None) -> str:
    """Used by the Streamlit 'chat with the resume database' feature."""
    if llm is None:
        llm = get_llm()
    prompt = (
        "You are a recruiting assistant answering a recruiter's follow-up question "
        "about a specific candidate, based ONLY on the resume text below. If the "
        "resume does not contain the answer, say so honestly instead of guessing.\n\n"
        f"Resume ({candidate['name']}):\n{candidate['text']}\n\n"
        f"Recruiter question: {question}\n\nAnswer:"
    )
    try:
        answer = llm(prompt)
    except Exception:
        answer = None
    if not answer:
        # simple keyword fallback
        q_terms = set(re.findall(r"[a-zA-Z]+", question.lower()))
        text = candidate["text"].lower()
        hits = [t for t in q_terms if t in text and len(t) > 3]
        if hits:
            answer = (
                f"[Rule-based fallback] The resume mentions related terms: {', '.join(hits)}. "
                f"Review the full text for exact context: {candidate['text'][:300]}..."
            )
        else:
            answer = (
                "[Rule-based fallback] The resume does not appear to explicitly mention "
                "this. Configure an LLM (OPENAI_API_KEY) for a more nuanced answer."
            )
    return answer
