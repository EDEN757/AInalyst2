#!/usr/bin/env python3
"""
Incremental SEC EDGAR Chunk, FAISS Embedder & Retriever

- One-time or incremental embed build.
- Append-only: skips already-indexed chunks.
- Provides functions to retrieve and assemble context for RAG queries.
"""
import os
import json
import logging
import gc
from dotenv import load_dotenv
import openai
import tiktoken
import faiss
import numpy as np

# Load OpenAI API key from .env or environment
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("OPENAI_API_KEY not set in environment or .env file")

# Configuration
EMBED_MODEL    = "text-embedding-3-small"
CHUNK_SIZE     = 1000
CHUNK_OVERLAP  = 200
DATA_DIR       = "data"
INDEX_FILE     = "faiss_index.idx"
METADATA_FILE  = "faiss_metadata.json"
BATCH_SIZE     = 100
K_RETRIEVE     = 5

# Memory management configuration
MAX_CHUNKS_IN_MEMORY = int(os.getenv("MAX_CHUNKS_IN_MEMORY", "1000"))  # Max chunks to hold in memory
SAVE_PROGRESS_EVERY = int(os.getenv("SAVE_PROGRESS_EVERY", "500"))     # Save index every N chunks

# Tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(tokenizer.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks


def embed_texts(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    resp = openai.embeddings.create(input=texts, model=model)
    return [d.embedding for d in resp.data]


def build_empty_faiss(dims: int) -> faiss.IndexIDMap:
    return faiss.IndexIDMap(faiss.IndexFlatIP(dims))


def initialize_index():
    """Load or create FAISS index and metadata."""
    if os.path.exists(METADATA_FILE) and os.path.exists(INDEX_FILE):
        logging.info("Loading existing metadata and index...")
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        existing_keys = {(m['ticker'], m['accession'], m['chunk_index']) for m in metadata}
        next_id = max(m['id'] for m in metadata) + 1
        index = faiss.read_index(INDEX_FILE)
    else:
        logging.info("No existing index found. Creating fresh.")
        metadata = []
        existing_keys = set()
        next_id = 0
        index = None
    return index, metadata, existing_keys, next_id


def save_index_metadata(index, metadata):
    faiss.write_index(index, INDEX_FILE)
    logging.info(f"FAISS index saved to {INDEX_FILE}.")
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logging.info(f"Metadata saved to {METADATA_FILE}.")


def process_chunk_batch(chunks_batch, entries_batch, index, metadata):
    """Process a batch of chunks: embed them and add to index."""
    if not chunks_batch:
        return index, metadata
    
    logging.info(f"Processing batch of {len(chunks_batch)} chunks...")
    
    # Generate embeddings for this batch
    batch_embeddings = []
    for i in range(0, len(chunks_batch), BATCH_SIZE):
        batch = chunks_batch[i:i+BATCH_SIZE]
        logging.info(f"  Embedding sub-batch {i//BATCH_SIZE + 1}: {len(batch)} chunks")
        batch_embeddings.extend(embed_texts(batch))
    
    # Build or extend index
    dims = len(batch_embeddings[0])
    if index is None:
        index = build_empty_faiss(dims)
    
    # Add embeddings to index
    arr = np.array(batch_embeddings, dtype='float32')
    faiss.normalize_L2(arr)
    ids = np.array([e['id'] for e in entries_batch], dtype='int64')
    index.add_with_ids(arr, ids)
    
    # Update metadata
    metadata.extend(entries_batch)
    
    logging.info(f"Added {len(entries_batch)} vectors to index. Total vectors: {index.ntotal}")
    
    # Force garbage collection to free memory
    del batch_embeddings, arr
    gc.collect()
    
    return index, metadata


def update_embeddings():
    """Main driver: find new chunks, embed, and append to FAISS with memory-efficient processing."""
    index, metadata, existing_keys, next_id = initialize_index()

    # Batch accumulators for memory-efficient processing
    current_chunks = []
    current_entries = []
    total_new_chunks = 0
    processed_chunks = 0

    logging.info(f"Starting memory-efficient embedding process:")
    logging.info(f"  - Max chunks in memory: {MAX_CHUNKS_IN_MEMORY}")
    logging.info(f"  - Save progress every: {SAVE_PROGRESS_EVERY} chunks")
    logging.info(f"  - Embedding batch size: {BATCH_SIZE}")

    # Walk through all filings in data/ - process one file at a time
    for ticker in os.listdir(DATA_DIR):
        tdir = os.path.join(DATA_DIR, ticker)
        if not os.path.isdir(tdir):
            continue
        
        for fname in os.listdir(tdir):
            if not fname.endswith('.json'):
                continue
            
            path = os.path.join(tdir, fname)
            
            # Process one file at a time to avoid loading all files into memory
            with open(path, 'r', encoding='utf-8') as f:
                record = json.load(f)
            
            accession = record.get('accession')
            filing_date = record.get('filing_date', '')
            full_text = record.get('text', '')

            # Split into chunks for this file only
            chunks = chunk_text(full_text)
            
            for idx, chunk in enumerate(chunks):
                key = (ticker, accession, idx)
                if key in existing_keys:
                    continue
                
                # Add to current batch
                current_chunks.append(chunk)
                current_entries.append({
                    'id': next_id,
                    'ticker': ticker,
                    'accession': accession,
                    'chunk_index': idx,
                    'filing_date': filing_date,
                    'form': record.get('form')
                })
                next_id += 1
                total_new_chunks += 1

                # Process batch when memory limit is reached
                if len(current_chunks) >= MAX_CHUNKS_IN_MEMORY:
                    index, metadata = process_chunk_batch(current_chunks, current_entries, index, metadata)
                    processed_chunks += len(current_chunks)
                    
                    # Clear memory
                    current_chunks.clear()
                    current_entries.clear()
                    
                    # Save progress periodically
                    if processed_chunks >= SAVE_PROGRESS_EVERY:
                        logging.info(f"Saving progress... ({processed_chunks} chunks processed)")
                        save_index_metadata(index, metadata)
                        processed_chunks = 0
            
            # Free memory after processing each file
            del record, chunks
            gc.collect()

    # Process remaining chunks
    if current_chunks:
        index, metadata = process_chunk_batch(current_chunks, current_entries, index, metadata)
        processed_chunks += len(current_chunks)

    if total_new_chunks == 0:
        logging.info("No new chunks to embed. Exiting.")
        return

    logging.info(f"Completed embedding {total_new_chunks} new chunks in memory-efficient batches.")
    save_index_metadata(index, metadata)


def load_chunk_text(entry: dict) -> str:
    """Given a metadata entry, re-load and return the exact chunk text."""
    path = os.path.join(DATA_DIR, entry['ticker'], f"{entry['accession']}.json")
    with open(path, 'r', encoding='utf-8') as f:
        record = json.load(f)
    chunks = chunk_text(record.get('text', ''))
    return chunks[entry['chunk_index']]


def retrieve(query: str, k: int = K_RETRIEVE) -> list[dict]:
    """Return top-k metadata entries for a query."""
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    index = faiss.read_index(INDEX_FILE)

    q_emb = openai.embeddings.create(input=[query], model=EMBED_MODEL).data[0].embedding
    arr = np.array([q_emb], dtype='float32')
    faiss.normalize_L2(arr)

    distances, ids = index.search(arr, k)
    hits = []
    for vid in ids[0]:
        if vid < 0:
            continue
        hits.append(metadata[vid])
    return hits


def answer_rag(query: str, k: int = K_RETRIEVE) -> str:
    """Fetch top-k chunks, assemble context, and call the chat model."""
    hits = retrieve(query, k)
    contexts = [load_chunk_text(h) for h in hits]
    context_blob = "\n\n---\n\n".join(contexts)
    messages = [
        {"role": "system", "content": "You are a financial assistant."},
        {"role": "user",   "content": f"Context:\n{context_blob}\n\nQ: {query}"}
    ]
    resp = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    return resp.choices[0].message.content


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    update_embeddings()
