<div align="center">

# 🧠 NLP Playground

**Five end-to-end NLP projects** — from classic TF-IDF pipelines to transformers and retrieval-augmented generation.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)

</div>

---

## 🚀 What's inside

| Folder | The one-liner | Best result |
|--------|----------------|-------------|
| [Task 1](#task-1--news-category-classifier) — News Category Classifier | Sort headlines into World / Sports / Business / Sci-Tech | **91.2%** accuracy |
| [Task 2](#task-2--text-summarization) — Text Summarization | Long articles → tight summaries | **0.44 ROUGE-1** |
| [Task 3](#task-3--question-answering) — Question Answering | Ask a question, get the exact answer span | **84.5 EM / 92.4 F1** |
| [Task 4](#task-4--sentiment-analysis) — Sentiment Analysis | Is this review a 👍 or a 👎? | **89.5%** accuracy |
| [Task 5](#task-5--rag-talent-search) — RAG Talent Search | "Find me a Junior Data Analyst who knows SQL" 🔍 | 220 resumes, live semantic search |

---

## Task 1 — News Category Classifier

> *120,000 AG News headlines, four categories, one model that has to pick.*

Full text pipeline (tokenize → stopwords → lemmatize) feeding TF-IDF into three head-to-head classifiers.

- 🥇 **Linear SVM: 91.2%** — edges out Logistic Regression (91.0%) and Random Forest (85.7%)
- 🎨 Bonus: word clouds per category + a Keras feedforward network (87.3%) for good measure

---

## Task 2 — Text Summarization

> *CNN/DailyMail articles, compressed to their essence by BART.*

- 📝 **ROUGE-1: 0.44** on abstractive summaries
- ⚖️ Bonus: extractive (TextRank) vs. abstractive showdown — abstractive wins, as the literature predicts
- 🔧 Bonus: fine-tuned T5-small as a taste of custom training

---

## Task 3 — Question Answering

> *Give it a paragraph and a question — it finds the exact words that answer it.*

Three transformer heavyweights, benchmarked on SQuAD v1.1:

- 🏆 **BERT-large (whole-word-masking): 84.5 EM / 92.4 F1**
- DistilBERT and RoBERTa close behind
- 💻 Bonus: an interactive CLI — paste your own passage, ask anything

---

## Task 4 — Sentiment Analysis

> *50,000 IMDb reviews walk into a classifier...*

Cleaned & vectorized (TF-IDF) reviews, then let **Logistic Regression** decide positive vs. negative.

- 🏆 **89.5% accuracy** (Naive Bayes bonus run: 86.4%)
- 📊 Bonus visualizations: the words that scream "positive" vs. "negative," plus the words the model itself leans on most

---

## Task 5 — RAG Talent Search

> *"Recruiters don't want to read 1,000 resumes." So this reads them instead.*

A full RAG pipeline over 220 real resumes:

`Embed → ChromaDB → semantic search → LLM explains the fit`

- 🎯 Ask in plain English, get ranked candidates *and* a written reason why each one fits
- 🆓 Runs free with a local model, or plug in an OpenAI key for sharper answers
- ⚖️ Bonus: an automated bias-check report + candidate follow-up Q&A

---

## 🛠️ Setup

```bash
pip install -r requirements.txt
```

Every notebook runs top-to-bottom on its own — no hidden setup steps, no missing pieces.
