import pymysql
import ollama

# -------------------
# Config
# -------------------
DB_HOST = "127.0.0.1"
DB_PORT = 4000
DB_USER = "root"
DB_PASS = "mypass123"
DB_NAME = "rag"

EMBED_MODEL = "nomic-embed-text:v1.5"   # embedding model
LLM_MODEL   = "gemma3:12b"              # LLM for final answers
TOP_K = 7                               # number of retrieved chunks

# -------------------
# Helpers
# -------------------
def vec_to_text(vec):
    # Convert list[float] to TiDB-friendly string format
    # Using 7 decimals since float32 has ~6–7 digits of precision
    return "[" + ",".join(f"{x:.7f}" for x in vec) + "]"

def embed_query(query: str):
    """Embed user question using Ollama"""
    resp = ollama.embeddings(model=EMBED_MODEL, prompt=query)
    return resp["embedding"]

def search_tidb(query_embedding):
    """Run vector search on TiDB"""
    conn = pymysql.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER,
        password=DB_PASS, database=DB_NAME, charset="utf8mb4"
    )
    cursor = conn.cursor()

    q_text = vec_to_text(query_embedding)

    sql = f"""
    SELECT id, title, page, chunk_type, content
    FROM documentation_chunks
    WHERE chunk_type IN ('text','code')  -- exclude 'index'
    ORDER BY VEC_COSINE_DISTANCE(embedding, VEC_FROM_TEXT(%s))
    LIMIT {TOP_K};
    """
    cursor.execute(sql, (q_text,))
    rows = cursor.fetchall()

    cursor.close()
    conn.close()
    return rows

def run_rag(question: str):
    """Embed, retrieve, and generate an answer with context"""
    print(f"\n🔎 Question: {question}\n")

    # Step 1: Embed question
    q_vec = embed_query(question)

    # Step 2: Retrieve from DB
    docs = search_tidb(q_vec)
    context = "\n\n".join([f"[{r[1]} - p.{r[2]}]\n{r[4]}" for r in docs])

    print("📖 Retrieved context:")
    for r in docs:
        print(f"- {r[1]} (page {r[2]})")

    # Step 3: Build prompt
    prompt = f"""You are a helpful assistant helping someone learn tidb.
Use the following documentation context to answer the question.

Context:
{context}

Question:
{question}

Answer clearly if possible reference the manual page as well, answer in english
"""

    # Step 4: Ask LLM (Streaming Mode)
    print("\n🤖 Answer:\n")
    stream = ollama.generate(model=LLM_MODEL, prompt=prompt, stream=True)
    for chunk in stream:
        print(chunk["response"], end="", flush=True)
    print("\n\n✅ Done.\n")

# -------------------
# Chat Loop
# -------------------
if __name__ == "__main__":
    print("💬 RAG Chatbot ready! Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit", "q"):
            print("👋 Goodbye!")
            break
        if not question:
            continue
        run_rag(question)
