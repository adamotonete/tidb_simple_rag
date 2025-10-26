# RAG with TiDB and Ollama Tutorial  

## 🎥 Video Walkthrough

You can watch the video version of this tutorial here:  
[▶ Watch on YouTube](https://youtu.be/0jhS3ABn-GU)

## Details:

This repository contains a step-by-step tutorial on how to build a **local Retrieval-Augmented Generation (RAG) chatbot** using:  

- **TiDB** with the `VECTOR` data type  
- **Ollama** for embeddings and LLM inference  
- **Python** for ingestion and querying scripts  

The tutorial shows how to:  
- Convert TiDB documentation into a **PDF** and split it into chunks  
- Store the text and embeddings in **TiDB**  
- Query the database with semantic search  
- Generate answers with **Gemma 12B** in a chatbot loop  

⚠️ **Note:**  
- This tutorial is intended as a **learning project**.  
- For simplicity, it does **not** use TiFlash ANN indexes, but mentions them as an option for production.  
- Running locally requires a **GPU with ≥12GB VRAM** or an **Apple Silicon Mac**.  

---

## Repository Contents  
- `tidb_docs_pdf.txt` → link to an already generated TiDB documentation in pdf
- `rag_tidb_ollama.html` → styled HTML version for publishing  
- `load.py` → script to split and embed documentation into TiDB  
- `chat.py` → chatbot script that retrieves context and answers questions  

---

## 🚀 Getting Started  

1. Install [TiUP](https://tiup.io/) and start TiDB Playground.  
2. Install [Ollama](https://ollama.ai) and pull the models:  
   ```bash
   ollama pull embeddinggemma
   ollama pull gemma:12b
3. Create a Python virtual environment and install requirements:
   ```
   python3 -m venv rag_tidb
   source rag_tidb/bin/activate
   pip install pymupdf pymysql ollama
   ```
4. Follow the tutorial in `rag_tidb_ollama.html`

## Author Note

I’m a database engineer, not an AI expert. This tutorial was created with AI assistance as part of my own learning.
There are probably better ways to implement RAG pipelines — feedback and improvements are very welcome!
   
