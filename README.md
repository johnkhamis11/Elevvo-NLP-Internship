# NLP Tasks — Transformer & Classical ML Pipelines

A collection of NLP mini-projects covering text classification, question answering, summarization, and retrieval-augmented generation (RAG), using both classical ML (TF-IDF + scikit-learn) and pre-trained transformer models (Hugging Face).

## 📁 Repository Structure

```
├── Task_1_Sentiment_Analysis_on_Product_Reviews.ipynb
├── Task_2_News_Category_Classification.ipynb
├── Task_6_Question_Answering_with_Transformers.ipynb
├── Task_7_Text_Summarization.ipynb
├── Task_8_RAG_Talent_Search.ipynb
├── rag_utils.py
└── requirements.txt
```

---

## Task 1 — Sentiment Analysis on Product Reviews

Binary sentiment classification (positive/negative) on the **IMDb Reviews** dataset (50,000 reviews, balanced).

**Pipeline:** text cleaning & stopword removal → TF-IDF vectorization → Logistic Regression

**Results:**
| Model | Accuracy |
|---|---|
| Logistic Regression | **89.47%** |
| Naive Bayes (bonus) | 86.44% |

**Bonus:** most frequent positive/negative words visualized, plus a plot of the most influential words per class based on model weights.

**Tools:** Pandas, NLTK, Scikit-learn

---

## Task 2 — News Category Classification

Multiclass classification of news articles into **World / Sports / Business / Sci-Tech** using the **AG News** dataset (120,000 train / 7,600 test articles).

**Pipeline:** tokenization → stopword removal → lemmatization → TF-IDF → multiclass classifiers

**Results:**
| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 91.0% | 0.910 |
| **Linear SVM** | **91.2%** | **0.912** |
| Random Forest | 85.7–86.1% | 0.856 |
| Feedforward NN (Keras) — bonus | 87.3% | — |

**Bonus:** top-word bar plots and word clouds per category; a simple feedforward neural network trained on TF-IDF features as an alternative to the classic models.

**Tools:** Pandas, Scikit-learn, NLTK, TensorFlow/Keras

---

## Task 6 — Question Answering with Transformers

Extractive QA system on **SQuAD v1.1**: given a context passage and a question, the model extracts the answer span.

**Models compared:**
| Model | Exact Match | F1 |
|---|---|---|
| DistilBERT (distilled-squad) | 82.5 | 91.87 |
| **BERT-large (whole-word-masking)** | **84.5** | **92.41** |
| RoBERTa-base (SQuAD2) | 76.5 | 86.56 |

**Bonus:** interactive command-line interface — enter a passage and ask questions about it, with an extracted answer + confidence score.

**Tools:** Hugging Face Transformers

---

## Task 7 — Text Summarization

Abstractive summarization of **CNN/DailyMail** news articles using **BART** (`facebook/bart-large-cnn`).

**ROUGE F1 scores (abstractive, BART):**
| Metric | Score |
|---|---|
| ROUGE-1 | 0.443 |
| ROUGE-2 | 0.233 |
| ROUGE-L | 0.320 |

**Bonus:**
- Extractive summarization with **TextRank** (`sumy`) for comparison — ROUGE-1: 0.310, ROUGE-2: 0.117, ROUGE-L: 0.200 (abstractive outperforms extractive, as expected).
- Fine-tuned **T5-small** on a small custom sample as a proof-of-concept fine-tuning workflow.

**Tools:** Hugging Face Transformers, Datasets, ROUGE-score

---

## Task 8 — RAG-Powered Talent Search Engine

A Retrieval-Augmented Generation pipeline that lets recruiters search resumes using natural-language queries (e.g. *"Junior Data Analyst who knows SQL and Tableau"*), built on the **Resume Entities for NER** dataset (220 real resumes).

**Pipeline:**
1. Parse resumes (DataTurks JSON annotation format) and extract skills/experience level.
2. Embed resumes with a HuggingFace sentence-transformer (`all-MiniLM-L6-v2`) and store them in **ChromaDB**.
3. Semantic search over the vector store given a natural-language query.
4. **LLM-based evaluation** (industry constraint): the top-3 retrieved resumes are passed to an LLM, which generates a text explanation of why each candidate fits — instead of a raw similarity score. Supports OpenAI/GPT if an API key is configured, otherwise falls back to a local HuggingFace model (`flan-t5-base`), so it runs fully free of charge.

**Bonus additions:**
- **Bias check** — flags skew in retrieval results by category, resume length, or (heavily caveated) name-based gender hints.
- Chat-style follow-up Q&A about a specific candidate.

**Tools:** LangChain, ChromaDB, Sentence-Transformers, Hugging Face Transformers, OpenAI (optional)

---

## Setup

```bash
pip install -r requirements.txt
```

Each notebook can be run end-to-end independently; datasets are loaded from local CSV/JSON files (or Hugging Face Datasets for Task 7).
