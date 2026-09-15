# Elevvo NLP Internship Tasks

This repository contains my completion of the 4 required NLP projects for the Elevvo Internship program.

## 📌 Projects Overview

### 1. Task 1: Sentiment Analysis
- **Model**: TF-IDF Vectorizer + Logistic Regression
- **Dataset**: IMDb Movie Reviews (Stanford NLP)
- **Performance**: ~82.8% Accuracy
- **Summary**: Text preprocessing, stop-word removal, and sentiment classification.

### 2. Task 6: Extractive Question Answering
- **Model**: DistilBERT (`distilbert-base-cased-distilled-squad`)
- **Summary**: Direct inference pipeline using HuggingFace Transformers to extract precise answer spans from contextual paragraphs.

### 3. Task 7: Generative Text Summarization
- **Model**: BART Large CNN (`facebook/bart-large-cnn`)
- **Summary**: Abstractive text summarization of tech articles using sequence-to-sequence neural architecture.

### 4. Task 8: RAG-Powered Talent Search Engine
- **Tech Stack**: LangChain, ChromaDB, Sentence Transformers (`all-MiniLM-L6-v2`), GPT-2
- **Summary**: End-to-end Retrieval-Augmented Generation system. Candidate resumes are embedded into a vector database for semantic similarity search, followed by LLM-driven response synthesis.

---
**Author**: John Khamis Samir
