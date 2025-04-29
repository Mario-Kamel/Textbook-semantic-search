import sys
sys.stdout.reconfigure(encoding='utf-8')

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# --- Step 1: Load the textbook ---
doc = fitz.open("networks.pdf")

# New: Keep track of which page each chunk comes from
page_texts = []
for page_number, page in enumerate(doc):
    text = page.get_text()
    page_texts.append((page_number + 1, text))  # 1-based page numbers

# --- Step 2: Split into chunks, keeping page info ---
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

chunks = []
chunk_pages = []

for page_num, text in page_texts:
    page_chunks = splitter.split_text(text)
    chunks.extend(page_chunks)
    chunk_pages.extend([page_num] * len(page_chunks))  # Keep page numbers aligned with chunks

# --- Step 3: Embed the chunks ---
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks, show_progress_bar=True)

# --- Step 4: Store in FAISS ---
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

# --- Step 5: Define semantic search ---
def search(query, k=5):
    query_vec = model.encode([query])
    D, I = index.search(query_vec, k)
    return [(chunks[i], chunk_pages[i], D[0][rank]) for rank, i in enumerate(I[0])]  # Return chunk, page, distance

# --- Step 6: Main loop to take user queries ---
print("\nSemantic Search Ready! 🔍 Type your question (or type 'exit' to quit)\n")

while True:
    user_query = input("❓ Your query: ")
    if user_query.lower() in ("exit", "quit"):
        print("Goodbye! 👋")
        break

    results = search(user_query, k=5)
    print("\nTop Results:\n")
    for text, page_num, score in results:
        print(f"📄 Page {page_num} | 🔢 Score: {score:.2f}")
        print(text.strip())
        print("\n" + "-" * 50 + "\n")
