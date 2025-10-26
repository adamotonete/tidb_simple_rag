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

EMBED_MODEL = "nomic-embed-text:v1.5"
LLM_MODEL   = "gemma3:12b"

TOP_K = 5   # number of results to retrieve

# -------------------
# Helpers
# -------------------
def embed_text(text: str):
    resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return resp["embedding"]

def vec_to_text(vec):
    return "[" + ",".join(f"{x:.7f}" for x in vec) + "]"

# -------------------
# Main
# -------------------
print("Connecting to TiDB...")
conn = pymysql.connect(
    host=DB_HOST, port=DB_PORT, user=DB_USER,
    password=DB_PASS, database=DB_NAME, charset="utf8mb4"
)
cursor = conn.cursor()

print("💬 RAG Chatbot ready! Type 'exit' to quit.")

# Ask language once
TARGET_LANGUAGE = input("\n🌍 In what language do you want the answers? (e.g., English, Portuguese, Spanish): ").strip()
if not TARGET_LANGUAGE:
    TARGET_LANGUAGE = "English"

while True:
    q = input("\nYou: ")
    if q.lower() in ["exit", "quit"]:
        break

    # 1. Embed query
    q_emb = embed_text(q)
    q_emb_text = vec_to_text(q_emb)

    # 2. Search DB using TiDB vector distance
    sql = f"""
        SELECT id, title, page, chunk_type, content
        FROM documentation_chunks
        WHERE chunk_type IN ('text','code')  -- exclude 'index'
        ORDER BY VEC_COSINE_DISTANCE(embedding, VEC_FROM_TEXT(%s))
        LIMIT {TOP_K};
    """
    cursor.execute(sql, (q_emb_text,))
    rows = cursor.fetchall()

    if not rows:
        print("🤔 No results found.")
        continue

    context = "\n\n".join(
        f"[Page {r[2]} - {r[1]}]\n{r[4]}" for r in rows if r[4]
    )

    # 3. Build prompt
    prompt = f"""You are a helpful assistant.
Use the following context from documentation to answer the question.
If you don't know, say you don't know.

Answer in {TARGET_LANGUAGE}.

Context:
{context}

Question: {q}
Answer:"""

    # 4. Ask LLM (streaming)
    print(f"\n🤖 Answer in {TARGET_LANGUAGE}:\n")
    stream = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        if "message" in chunk and "content" in chunk["message"]:
            print(chunk["message"]["content"], end="", flush=True)
    print("\n✅ Done.")
