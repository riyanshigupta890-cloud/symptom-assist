import json
import hashlib
"""
rag_pipeline.py
---------------
RAG (Retrieval-Augmented Generation) pipeline for the symptom chatbot.

Medical documents are loaded from data/medical_docs.csv.
Uses sentence-transformers if available, otherwise falls back to TF-IDF.
"""

import csv
import math
import os
import re
from collections import defaultdict
from .pubmed_retriever import PubMedRetriever

# ---------------------------------------------------------------------------
# 1. CSV Loader
# ---------------------------------------------------------------------------

def load_documents_from_csv(csv_path: str) -> list[dict]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Medical docs CSV not found: {csv_path}")

    documents: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            documents.append({
                "id":        f"doc_{row['condition'].strip()}_{i}",
                "condition": row["condition"].strip(),
                "title": (row.get("title") or "").strip(),
                "content": (row.get("content") or "").strip(),
            })

    print(f"[RAG] Loaded {len(documents)} medical documents from CSV")
    return documents


# ---------------------------------------------------------------------------
# 2. Lightweight TF-IDF Retriever (free-tier fallback)
# ---------------------------------------------------------------------------

class TFIDFRetriever:
    def __init__(self):
        self.documents = []
        self.tfidf_matrix = []
        self.vocab = {}

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z]+", text.lower())

    def _compute_tf(self, tokens: list[str]) -> dict:
        tf = defaultdict(float)
        for t in tokens:
            tf[t] += 1
        total = len(tokens) or 1
        return {k: v / total for k, v in tf.items()}

    def index(self, documents: list[dict], **kwargs):
        self.documents = documents
        all_tokens = []
        doc_tokens = []
        for doc in documents:
            tokens = self._tokenize(f"{doc.get('title','')} {doc.get('content','')}")
            doc_tokens.append(tokens)
            all_tokens.extend(set(tokens))

        # Build vocab
        vocab = sorted(set(all_tokens))
        self.vocab = {w: i for i, w in enumerate(vocab)}
        N = len(documents)

        # Document frequency
        df = defaultdict(int)
        for tokens in doc_tokens:
            for w in set(tokens):
                df[w] += 1

        # TF-IDF matrix
        self.tfidf_matrix = []
        for tokens in doc_tokens:
            tf = self._compute_tf(tokens)
            vec = {}
            for w, tf_val in tf.items():
                idf = math.log((N + 1) / (df[w] + 1)) + 1
                vec[w] = tf_val * idf
            self.tfidf_matrix.append(vec)

        print(f"[RAG] TF-IDF index built: {N} documents, {len(vocab)} terms")

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        q_tokens = self._tokenize(query)
        q_tf = self._compute_tf(q_tokens)

        scores = []
        for i, doc_vec in enumerate(self.tfidf_matrix):
            score = sum(q_tf.get(w, 0) * doc_vec.get(w, 0) for w in q_tf)
            scores.append((score, i))

        scores.sort(reverse=True)
        results = []
        for score, idx in scores[:top_k]:
            doc = self.documents[idx]
            results.append({
                "id": doc["id"],
                "original_id": doc["id"],
                "condition": doc["condition"],
                "title": doc.get("title", ""),
                "content": doc.get("content", ""),
                "relevance_score": round(score, 4)
            })
        return results


# ---------------------------------------------------------------------------
# 3. Semantic Retriever (uses sentence-transformers + chromadb if available)
# ---------------------------------------------------------------------------

class SemanticRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", db_path: str = None, model=None):
        try:
            from sentence_transformers import SentenceTransformer
            import chromadb
            import pathlib

            if model is not None:
                self.model = model
            else:
                self.model = SentenceTransformer(model_name)

            if db_path is None:
                _here = pathlib.Path(__file__).parent.parent.parent
                db_path = str(_here / "data" / "chroma_db")

            os.makedirs(db_path, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=db_path)
            self.collection = self.chroma_client.get_or_create_collection(name="medical_docs")
            self._use_semantic = True
            print("[RAG] Using semantic retriever (sentence-transformers)")
        except Exception as e:
            print(f"[RAG] sentence-transformers unavailable ({e}), falling back to TF-IDF")
            self._use_semantic = False
            self._tfidf = TFIDFRetriever()

    def _chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
        words = text.split()
        if not words:
            return []
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i: i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
        return chunks

    def index(self, documents: list[dict], chunk_size: int = 400, overlap: int = 50):
        if not self._use_semantic:
            self._tfidf.index(documents)
            return

        csv_hash = hashlib.md5(
            json.dumps([d["id"] for d in documents], sort_keys=True).encode()
        ).hexdigest()

        stored_hash = (self.collection.metadata or {}).get("csv_hash")

        if self.collection.count() > 0 and stored_hash == csv_hash:
            print(f"[RAG] ChromaDB up to date ({self.collection.count()} chunks). Skipping indexing.")
            return
        elif self.collection.count() > 0:
            print("[RAG] CSV changed — rebuilding index...")
            self.chroma_client.delete_collection("medical_docs")
            self.collection = self.chroma_client.get_or_create_collection(
                name="medical_docs",
                metadata={"csv_hash": csv_hash}
            )

        ids, texts_to_embed, metadatas = [], [], []
        seen_ids = set()

        for doc in documents:
            content = doc.get("content", "")
            title = doc.get("title", "")
            doc_chunks = self._chunk_text(content, chunk_size, overlap)

            for i, chunk_text in enumerate(doc_chunks):
                chunk_id = f"{doc['id']}_chunk_{i}"
                if chunk_id in seen_ids:
                    continue
                seen_ids.add(chunk_id)
                embed_text = f"{title}: {chunk_text}"
                ids.append(chunk_id)
                texts_to_embed.append(embed_text)
                metadatas.append({
                    "original_id": doc["id"],
                    "condition": doc["condition"],
                    "title": title,
                    "content": chunk_text
                })

        if texts_to_embed:
            print(f"[RAG] Generating embeddings for {len(texts_to_embed)} chunks...")
            embeddings = self.model.encode(texts_to_embed).tolist()
            self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts_to_embed)
            print("[RAG] Indexing complete.")

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        if not self._use_semantic:
            return self._tfidf.retrieve(query, top_k=top_k)

        if self.collection.count() == 0:
            return []

        query_embedding = self.model.encode(query).tolist()
        fetch_k = top_k * 3
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            include=["metadatas", "distances"]
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        final_results = []
        seen_docs = set()
        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i in range(len(ids)):
            if len(final_results) >= top_k:
                break
            dist = distances[i]
            score = max(0.0, 1.0 - (dist / 2.0))
            meta = metadatas[i]
            doc_id = meta["original_id"]
            if doc_id not in seen_docs:
                final_results.append({
                    "id": ids[i],
                    "original_id": doc_id,
                    "condition": meta["condition"],
                    "title": meta["title"],
                    "content": meta["content"],
                    "relevance_score": round(score, 4)
                })
                seen_docs.add(doc_id)

        return final_results


# ---------------------------------------------------------------------------
# 4. RAG Pipeline class
# ---------------------------------------------------------------------------

class RAGPipeline:
    def __init__(self, csv_path: str | None = None):
        if csv_path and os.path.exists(csv_path):
            documents = load_documents_from_csv(csv_path)
        else:
            documents = []
            print("[RAG] WARNING: no medical_docs.csv found; RAG context will be empty.")

        self.retriever = SemanticRetriever()
        self.retriever.index(documents)
        self.pubmed = PubMedRetriever()

    def retrieve_pubmed_raw(self, condition: str, query: str, top_k: int = 3) -> list[dict]:
        abstracts = self.pubmed.retrieve(condition, max_results=5)
        if not abstracts:
            return []

        docs = []
        for a in abstracts:
            docs.append({
                "id": f"pubmed_{a['pmid']}",
                "condition": condition,
                "title": f"{a['title']} [PMID: {a['pmid']}, {a['year']}]",
                "content": a['abstract']
            })

        temp_retriever = SemanticRetriever(model=getattr(self.retriever, 'model', None))
        temp_retriever.index(docs, chunk_size=400, overlap=50)
        return temp_retriever.retrieve(query, top_k=top_k)

    def retrieve_pubmed_context(self, condition: str, query: str, top_k: int = 3) -> str:
        docs = self.retrieve_pubmed_raw(condition, query, top_k=top_k)
        if not docs:
            return ""
        parts = [f"[{doc['title']}]\n{doc['content']}" for doc in docs]
        return "\n\n---\n\n".join(parts)

    def retrieve_context(self, query: str, top_k: int = 3, min_score: float = 0.3) -> str:
        docs = self.retriever.retrieve(query, top_k=top_k)
        if not docs:
            return ""
        docs = [doc for doc in docs if doc["relevance_score"] >= min_score]
        if not docs:
            return ""
        parts = [f"[{doc['title']}]\n{doc['content']}" for doc in docs]
        return "\n\n---\n\n".join(parts)

    def retrieve_raw(self, query: str, top_k: int = 3) -> list[dict]:
        return self.retriever.retrieve(query, top_k=top_k)


# ---------------------------------------------------------------------------
# 5. Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pathlib
    _here = pathlib.Path(__file__).parent.parent.parent
    csv_p = str(_here / "data" / "medical_docs.csv")

    rag = RAGPipeline(csv_path=csv_p)
    for q in [
        "I have a headache that throbs on one side with light sensitivity",
        "burning when I urinate and need to go frequently",
        "stomach cramps and vomiting after eating out",
    ]:
        print(f"\nQuery: '{q}'")
        for r in rag.retrieve_raw(q, top_k=2):
            print(f"  -> {r['title']} (score: {r['relevance_score']})")
