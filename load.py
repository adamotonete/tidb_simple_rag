import fitz  # PyMuPDF
import re
import pymysql
import ollama
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------
# Config
# -------------------
PDF_PATH = "output.pdf" #replace it here
CHUNK_SIZE = 1000
CHUNK_OVERLAP = int(CHUNK_SIZE * 0.3)  # 30% overlap
BATCH_PAGES = 20
MAX_WORKERS = 8
MIN_CHUNK_LEN = 200   # very small chunks are usually noise can be something like 80 as well 

DB_HOST = "127.0.0.1"
DB_PORT = 4000
DB_USER = "root"
DB_PASS = "mypass123"
DB_NAME = "rag"

EMBED_MODEL = "nomic-embed-text:v1.5"  # embeddings

# -------------------
# Helpers
# -------------------
def clean_text(text: str) -> str:
    """Normalize text by collapsing multiple spaces/newlines."""
    return re.sub(r"\s+", " ", text).strip()

def is_code_block(text: str) -> bool:
    code_keywords = ["CREATE TABLE", "SELECT", "INSERT", "UPDATE", "DELETE",
                     "function", "class", "{", "}", ";", "=>"]
    if text.strip().startswith(("    ", "\t")):
        return True
    return any(kw in text for kw in code_keywords)

def classify_chunk(text: str, block_type: str):
    """Classify text as text/code/index/TOC (skip TOC)."""
    stripped = text.strip()

    # Rule 1: Skip Table of Contents entries
    if (
        stripped.count('.') >= 5
        or stripped.count('·') >= 5
        or re.match(r".*\s\d{1,4}\s*$", stripped)
    ):
        return "toc"

    # Rule 2: Very short = index noise
    if len(stripped) < MIN_CHUNK_LEN:
        return "index"

    return block_type

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(clean_text(chunk))
        start += max(1, chunk_size - overlap)
    return chunks

def embed_text_ollama(text: str):
    resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return resp["embedding"]

def vec_to_text(vec):
    return "[" + ",".join(f"{x:.7f}" for x in vec) + "]"

def process_chunk(section_title, page_num, block_type, chunk):
    chunk = clean_text(chunk)
    emb = embed_text_ollama(chunk)
    emb_text = vec_to_text(emb)
    sql = """
        INSERT INTO documentation_chunks
        (title, page, chunk_type, content, embedding)
        VALUES (%s, %s, %s, %s, VEC_FROM_TEXT(%s))
    """
    return sql, (section_title, page_num, block_type, chunk, emb_text)

# -------------------
# Main
# -------------------
print("Loading PDF...")
doc = fitz.open(PDF_PATH)
total_pages = len(doc)

# Detect embedding dimension
print("Checking embedding dimension...")
test_emb = embed_text_ollama("test")
DIM = len(test_emb)
print(f"Embedding model {EMBED_MODEL} has dimension: {DIM}")

print("Connecting to TiDB...")
conn = pymysql.connect(
    host=DB_HOST, port=DB_PORT, user=DB_USER,
    password=DB_PASS, database=DB_NAME, charset="utf8mb4", autocommit=False
)
cursor = conn.cursor()

# Ensure table exists
cursor.execute(f"""
CREATE TABLE IF NOT EXISTS documentation_chunks (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    title VARCHAR(255),
    page INT,
    chunk_type ENUM('text','code','index'),
    content TEXT,
    embedding VECTOR({DIM})
);
""")

toc = doc.get_toc(simple=True)
section_map = {p: t for _, t, p in toc}

start_time = time.time()

print("Processing pages...")
for page_num, page in enumerate(doc, start=1):
    blocks = page.get_text("blocks")
    section_title = section_map.get(page_num, "Unknown Section")

    tasks = []
    for b in blocks:
        text = clean_text(b[4])
        if not text:
            continue

        block_type = "code" if is_code_block(text) else "text"
        block_type = classify_chunk(text, block_type)

        if block_type == "toc":
            continue

        for chunk in chunk_text(text):
            block_type_final = classify_chunk(chunk, block_type)
            if block_type_final == "toc":
                continue
            if block_type_final == "index":
                cursor.execute(
                    """
                    INSERT INTO documentation_chunks
                    (title, page, chunk_type, content, embedding)
                    VALUES (%s, %s, %s, %s, NULL)
                    """,
                    (section_title, page_num, "index", chunk)
                )
            else:
                tasks.append((section_title, page_num, block_type_final, chunk))

    # Process useful chunks in parallel
    if tasks:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_chunk, *t) for t in tasks]
            for f in as_completed(futures):
                try:
                    sql, params = f.result()
                    cursor.execute(sql, params)
                except Exception as e:
                    print(f"❌ Error on page {page_num}: {e}")

    # --- Progress & Commit ---
    if page_num % BATCH_PAGES == 0 or page_num == total_pages:
        conn.commit()
        pct = (page_num / total_pages) * 100
        elapsed = time.time() - start_time
        avg_time = elapsed / page_num
        eta = avg_time * (total_pages - page_num)
        print(f"✅ Progress: {page_num}/{total_pages} pages ({pct:.2f}%) "
              f"- elapsed {elapsed/60:.1f} min, ETA {eta/60:.1f} min")

# Final commit
conn.commit()
cursor.close()
conn.close()

print("🎉 Done: All pages processed and saved in TiDB (TOC entries skipped, spaces normalized).")
