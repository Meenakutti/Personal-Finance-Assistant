"""
RAG (Retrieval-Augmented Generation) utilities for the Personal Finance Assistant.
"""

import os
import json
import time
import pickle
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod
from pathlib import Path
from utils.trace_logger import get_tracer

_tracer = get_tracer(__name__)

# Singleton registry — one RAGPipeline instance per vector_store_type
_instances: Dict[str, "RAGPipeline"] = {}


class VectorStoreBase(ABC):
    """Base class for vector store implementations."""

    @abstractmethod
    def add_documents(self, documents: List[Dict[str, str]], metadatas: Optional[List[Dict]] = None):
        """Add documents to the vector store."""
        pass

    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        pass

    @abstractmethod
    def delete_all(self):
        """Clear all documents from the store."""
        pass


class SimpleVectorStore(VectorStoreBase):
    """Simple in-memory vector store implementation without external dependencies."""

    def __init__(self, collection_name: str = "finance_knowledge"):
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.collection_name = collection_name
        self.doc_counter = 0

    def add_documents(self, documents: List[Dict[str, str]], metadatas: Optional[List[Dict]] = None):
        """Add documents to the store."""
        for i, doc in enumerate(documents):
            doc_id = f"doc_{self.doc_counter}_{i}"
            self.documents[doc_id] = {
                "content": doc.get("content", ""),
                "metadata": metadatas[i] if metadatas and i < len(metadatas) else {},
                "id": doc.get("id", doc_id),
                "title": doc.get("title", ""),
                "category": doc.get("category", ""),
                "source": doc.get("source", "")
            }
        self.doc_counter += 1

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search documents by keyword matching."""
        query_words = set(query.lower().split())
        scores = []
        
        for doc_id, doc in self.documents.items():
            # Simple keyword matching
            content = (doc.get("content", "") + " " + doc.get("title", "")).lower()
            content_words = set(content.split())
            
            # Calculate overlap score
            overlap = len(query_words & content_words)
            if overlap > 0:
                scores.append((overlap, doc_id, doc))
        
        # Sort by score and return top k
        scores.sort(reverse=True, key=lambda x: x[0])
        results = []
        for score, doc_id, doc in scores[:k]:
            results.append({
                "content": doc.get("content", ""),
                "metadata": doc.get("metadata", {}),
                "id": doc["id"],
                "title": doc.get("title", ""),
                "category": doc.get("category", ""),
                "source": doc.get("source", ""),
                "distance": 1.0 - (score / (len(query_words) + 1))
            })
        return results

    def delete_all(self):
        """Clear all documents."""
        self.documents.clear()
        self.doc_counter = 0


class ChromaVectorStore(VectorStoreBase):
    """ChromaDB vector store implementation with fallback to simple store."""

    def __init__(self, collection_name: str = "finance_knowledge"):
        self.use_chroma = False
        self.simple_store = SimpleVectorStore(collection_name)
        
        try:
            import chromadb
            # Initialize with ephemeral client (in-memory)
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self.use_chroma = True
        except Exception as e:
            print(f"Note: ChromaDB unavailable ({type(e).__name__}). Using simple vector store instead.")
            self.use_chroma = False

    def add_documents(self, documents: List[Dict[str, str]], metadatas: Optional[List[Dict]] = None):
        """Add documents to Chroma or simple store."""
        if self.use_chroma:
            try:
                ids = [f"doc_{i}" for i in range(len(documents))]
                contents = [doc.get("content", "") for doc in documents]
                self.collection.add(
                    ids=ids,
                    documents=contents,
                    metadatas=metadatas or [{"source": doc.get("source", "unknown")} for doc in documents]
                )
            except Exception as e:
                print(f"Error adding to ChromaDB: {e}. Using simple store.")
                self.use_chroma = False
                self.simple_store.add_documents(documents, metadatas)
        else:
            self.simple_store.add_documents(documents, metadatas)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search Chroma collection or simple store."""
        if self.use_chroma:
            try:
                results = self.collection.query(query_texts=[query], n_results=k)
                docs = []
                if results and results["documents"]:
                    for i, doc in enumerate(results["documents"][0]):
                        docs.append({
                            "content": doc,
                            "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                            "distance": results["distances"][0][i] if results["distances"] else 0
                        })
                return docs
            except Exception as e:
                print(f"Error querying ChromaDB: {e}. Using simple store.")
                self.use_chroma = False
                return self.simple_store.search(query, k)
        else:
            return self.simple_store.search(query, k)

    def delete_all(self):
        """Delete all documents from collection."""
        if self.use_chroma:
            try:
                # For ChromaDB 0.4.x, we need to delete the collection
                pass
            except:
                pass
        self.simple_store.delete_all()


class FAISSVectorStore(VectorStoreBase):
    """FAISS-based vector store for semantic similarity search."""

    def __init__(self, embed_fn=None, batch_embed_fn=None,
                 collection_name: str = "finance_knowledge"):
        self.embed_fn = embed_fn
        self.batch_embed_fn = batch_embed_fn  # preferred: one API call for all docs
        self.collection_name = collection_name
        self.documents: List[Dict[str, Any]] = []
        self.index = None
        self.dimension: Optional[int] = None
        self.fallback = SimpleVectorStore(collection_name)

        try:
            import faiss  # type: ignore
            self.faiss = faiss
            self.use_faiss = True
        except ImportError:
            print("Note: faiss not installed, falling back to simple store. Install with: pip install faiss-cpu")
            self.use_faiss = False

    def add_documents(self, documents: List[Dict[str, str]], metadatas: Optional[List[Dict]] = None):
        """Embed and index documents with FAISS, or fall back to keyword store.

        Uses batch_embed_fn (one API call) when available; falls back to
        per-document embed_fn only when no batch function is provided.
        """
        if not self.use_faiss or (self.embed_fn is None and self.batch_embed_fn is None):
            self.fallback.add_documents(documents, metadatas)
            return

        import numpy as np

        contents = [doc.get("content", "") for doc in documents]

        # ── batch embed (preferred: 1 API call for all docs) ──────────────────
        if self.batch_embed_fn is not None:
            embeddings = self.batch_embed_fn(contents)
            if embeddings is None:
                self.fallback.add_documents(documents, metadatas)
                return
        else:
            # per-document fallback
            embeddings = []
            for content in contents:
                emb = self.embed_fn(content)
                if emb is None:
                    self.fallback.add_documents(documents, metadatas)
                    return
                embeddings.append(emb)

        valid_docs = [
            {
                "content": contents[i],
                "metadata": metadatas[i] if metadatas and i < len(metadatas) else {},
                "id": doc.get("id", f"doc_{len(self.documents) + i}"),
                "title": doc.get("title", ""),
                "category": doc.get("category", ""),
                "source": doc.get("source", ""),
            }
            for i, doc in enumerate(documents)
        ]

        emb_array = np.array(embeddings, dtype=np.float32)

        if self.index is None:
            self.dimension = emb_array.shape[1]
            self.index = self.faiss.IndexFlatIP(self.dimension)

        self.faiss.normalize_L2(emb_array)
        self.index.add(emb_array)
        self.documents.extend(valid_docs)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Semantic search via FAISS inner-product (cosine) similarity."""
        if not self.use_faiss or self.embed_fn is None or self.index is None or self.index.ntotal == 0:
            return self.fallback.search(query, k)

        import numpy as np

        query_emb = self.embed_fn(query)
        if query_emb is None:
            return self.fallback.search(query, k)

        query_array = np.array([query_emb], dtype=np.float32)
        self.faiss.normalize_L2(query_array)

        scores, indices = self.index.search(query_array, min(k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = self.documents[idx]
            results.append({
                "content": doc["content"],
                "metadata": doc["metadata"],
                "id": doc["id"],
                "title": doc["title"],
                "category": doc["category"],
                "source": doc["source"],
                "distance": float(1.0 - score),
            })
        return results

    def save(self, path: str) -> bool:
        """Persist the FAISS index and document store to disk."""
        if not self.use_faiss or self.index is None:
            return False
        try:
            import numpy as np
            payload = {
                "documents": self.documents,
                "dimension": self.dimension,
                "index_bytes": self.faiss.serialize_index(self.index).tobytes(),
            }
            with open(path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            return True
        except Exception as e:
            _tracer.warn("faiss_save_failed", error=str(e), path=path)
            return False

    def load(self, path: str) -> bool:
        """Load a previously persisted FAISS index from disk."""
        if not self.use_faiss or not os.path.exists(path):
            return False
        try:
            import numpy as np
            with open(path, "rb") as f:
                payload = pickle.load(f)
            import faiss  # type: ignore
            index_bytes = np.frombuffer(payload["index_bytes"], dtype=np.uint8)
            self.index = faiss.deserialize_index(index_bytes)
            self.documents = payload["documents"]
            self.dimension = payload["dimension"]
            return True
        except Exception as e:
            _tracer.warn("faiss_load_failed", error=str(e), path=path)
            return False

    def delete_all(self):
        """Clear the FAISS index and document store."""
        self.documents = []
        self.index = None
        self.dimension = None
        self.fallback.delete_all()


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline (singleton per vector_store_type)."""

    def __new__(cls, vector_store_type: str = "simple"):
        if vector_store_type not in _instances:
            _instances[vector_store_type] = super().__new__(cls)
        return _instances[vector_store_type]

    def __init__(self, vector_store_type: str = "simple"):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        # Initialize embeddings first so FAISS store can receive the embed_fn
        self.embeddings = self._initialize_embeddings_model()
        self.knowledge_base_loaded = False
        self._query_embed_cache: Dict[str, Optional[List[float]]] = {}

        if vector_store_type == "simple":
            self.vector_store = SimpleVectorStore()
        elif vector_store_type == "chroma":
            self.vector_store = ChromaVectorStore()
        elif vector_store_type == "faiss":
            self.vector_store = FAISSVectorStore(
                embed_fn=self._get_embedding_cached,
                batch_embed_fn=self._get_embeddings_batch,
            )
        else:
            raise ValueError(f"Unsupported vector store: {vector_store_type}")

    def _initialize_embeddings_model(self):
        """Initialize OpenAI embeddings model for vector operations."""
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                _tracer.warn("embeddings_init_skipped", reason="OPENAI_API_KEY not set")
                return None
            embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            client = OpenAI(api_key=api_key)
            _tracer.step("embeddings_initialized", model=embedding_model)
            return {"client": client, "model": embedding_model, "type": "openai"}
        except ImportError:
            _tracer.warn("embeddings_init_skipped", reason="openai package not installed")
            return None
        except Exception as e:
            _tracer.error("embeddings_init_failed", error=str(e))
            return None

    def _index_cache_path(self) -> str:
        """Disk path for the persisted FAISS index.
        Override with FAISS_INDEX_PATH env var (e.g. a Docker volume mount)."""
        env_path = os.getenv("FAISS_INDEX_PATH")
        if env_path:
            return env_path
        return os.path.join(
            os.path.dirname(__file__), "..", "knowledge_base", "faiss_index.pkl"
        )

    def _get_embeddings_batch(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Embed a list of texts in a single OpenAI API call (batch embed)."""
        if not self.embeddings or self.embeddings.get("type") != "openai":
            return None
        try:
            client = self.embeddings["client"]
            model = self.embeddings["model"]
            t0 = time.perf_counter()
            response = client.embeddings.create(input=texts, model=model)
            embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
            _tracer.timing("batch_embed", time.perf_counter() - t0,
                           texts=len(texts), model=model)
            return embeddings
        except Exception as e:
            _tracer.warn("batch_embed_failed", error=str(e))
            return None

    def _get_embedding_cached(self, text: str) -> Optional[List[float]]:
        """Embed a single query string with an in-process LRU cache."""
        if text in self._query_embed_cache:
            _tracer.detail("query_embed_cache_hit", text_len=len(text))
            return self._query_embed_cache[text]
        result = self._get_embedding(text)
        # Keep cache bounded to 256 entries
        if len(self._query_embed_cache) >= 256:
            self._query_embed_cache.pop(next(iter(self._query_embed_cache)))
        self._query_embed_cache[text] = result
        return result

    def load_knowledge_base(self, knowledge_base_path: Optional[str] = None):
        """Load documents from JSON knowledge base file into vector store.

        Fast path: load a persisted FAISS index from disk (< 1s).
        Slow path: batch-embed all docs in one API call, then persist for next run.
        """
        if knowledge_base_path is None:
            knowledge_base_path = os.path.join(
                os.path.dirname(__file__), "..", "knowledge_base",
                "financial_knowledge_base.json"
            )

        if not os.path.exists(knowledge_base_path):
            _tracer.warn("kb_file_missing", path=knowledge_base_path)
            return False

        # ── fast path: load persisted FAISS index ─────────────────────────────
        if isinstance(self.vector_store, FAISSVectorStore):
            index_path = self._index_cache_path()
            t0 = time.perf_counter()
            if self.vector_store.load(index_path):
                _tracer.timing("kb_loaded_from_disk", time.perf_counter() - t0,
                               docs=len(self.vector_store.documents), path=index_path)
                self.knowledge_base_loaded = True
                return True

        # ── slow path: embed + index from scratch ─────────────────────────────
        try:
            with open(knowledge_base_path, "r") as f:
                documents = json.load(f)
            _tracer.step("kb_raw_loaded", doc_count=len(documents),
                         path=knowledge_base_path)

            t0 = time.perf_counter()
            parsed_documents = self._parse_and_chunk_documents(documents)
            _tracer.timing("kb_chunked", time.perf_counter() - t0,
                           chunks=len(parsed_documents), source_docs=len(documents))

            t0 = time.perf_counter()
            self._add_documents_to_store(parsed_documents)
            _tracer.timing("kb_indexed", time.perf_counter() - t0,
                           chunks=len(parsed_documents),
                           store=type(self.vector_store).__name__)

            # Persist for subsequent runs
            if isinstance(self.vector_store, FAISSVectorStore):
                index_path = self._index_cache_path()
                if self.vector_store.save(index_path):
                    _tracer.step("kb_index_persisted", path=index_path,
                                 chunks=len(parsed_documents))

            self.knowledge_base_loaded = True
            _tracer.step("kb_ready", chunks=len(parsed_documents))
            return True

        except Exception as e:
            _tracer.error("kb_load_failed", error=str(e))
            return False

    def _parse_and_chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Parse and chunk documents for indexing."""
        chunked_documents = []
        
        for doc in documents:
            # Extract document metadata
            doc_id = doc.get("id", "")
            category = doc.get("category", "")
            title = doc.get("title", "")
            content = doc.get("content", "")
            source = doc.get("source", "Unknown")
            
            # Create full text with metadata
            full_text = f"{title}\n{content}"
            
            # For simple documents, create single chunk
            # In production, would split large documents into smaller chunks
            chunk_size = 500  # words
            words = full_text.split()
            
            if len(words) <= chunk_size:
                # Single chunk
                chunked_documents.append({
                    "id": doc_id,
                    "content": full_text,
                    "metadata": {
                        "category": category,
                        "source": source,
                        "title": title,
                        "chunk": 1,
                        "total_chunks": 1
                    }
                })
            else:
                # Split into multiple chunks with overlap
                overlap_words = 50
                chunk_words = chunk_size
                start = 0
                chunk_num = 1
                total_chunks = (len(words) + chunk_size - overlap_words - 1) // (chunk_size - overlap_words)
                
                while start < len(words):
                    end = min(start + chunk_words, len(words))
                    chunk_content = " ".join(words[start:end])
                    
                    chunked_documents.append({
                        "id": f"{doc_id}_chunk_{chunk_num}",
                        "content": chunk_content,
                        "metadata": {
                            "category": category,
                            "source": source,
                            "title": title,
                            "chunk": chunk_num,
                            "total_chunks": total_chunks
                        }
                    })
                    
                    start = end - overlap_words
                    chunk_num += 1
        
        return chunked_documents

    def _add_documents_to_store(self, documents: List[Dict[str, str]]):
        """Add parsed and chunked documents to vector store."""
        if not documents:
            return
        
        # Prepare documents and metadata for vector store
        doc_list = []
        metadata_list = []
        
        for doc in documents:
            doc_list.append({
                "id": doc.get("id", ""),
                "content": doc.get("content", "")
            })
            metadata_list.append(doc.get("metadata", {}))
        
        # Add to vector store
        self.vector_store.add_documents(doc_list, metadata_list)

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text using OpenAI API."""
        if not self.embeddings or self.embeddings.get("type") != "openai":
            return None
        
        try:
            client = self.embeddings["client"]
            model = self.embeddings["model"]
            
            # Call OpenAI embedding API
            response = client.embeddings.create(
                input=text,
                model=model
            )
            
            # Extract embedding from response
            embedding = response.data[0].embedding
            return embedding
            
        except Exception as e:
            print(f"Error getting embedding from OpenAI: {e}")
            return None

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        if not self.knowledge_base_loaded:
            _tracer.step("kb_lazy_load")
            self.load_knowledge_base()

        # Short-circuit: if the FAISS index has no documents or embeddings are not
        # configured, skip straight to keyword fallback — avoids a wasted API call.
        if isinstance(self.vector_store, FAISSVectorStore) and (
            self.vector_store.index is None
            or self.vector_store.index.ntotal == 0
            or self.embeddings is None
        ):
            _tracer.detail("faiss_not_ready_keyword_fallback",
                           has_index=self.vector_store.index is not None,
                           has_embeddings=self.embeddings is not None)
            return self.vector_store.fallback.search(query, k)

        t0 = time.perf_counter()
        results = self.vector_store.search(query, k=k)
        _tracer.timing("rag_search", time.perf_counter() - t0,
                       k=k, returned=len(results),
                       store=type(self.vector_store).__name__)
        if results:
            categories = list({r.get("metadata", {}).get("category", "?") for r in results})
            _tracer.detail("rag_results", categories=categories,
                           top_score=round(1 - results[0].get("distance", 1), 3) if results else None)
        return results

    def augment_query(self, query: str, k: int = 5) -> str:
        """Augment query with retrieved context."""
        retrieved_docs = self.retrieve(query, k=k)

        context = "\n\n".join([
            f"Source: {doc['metadata'].get('source', 'Unknown')}\nCategory: {doc['metadata'].get('category', 'N/A')}\n{doc['content']}"
            for doc in retrieved_docs
        ])

        augmented = f"""
Context from Financial Knowledge Base:
{context}

User Query: {query}
"""
        return augmented

    def add_documents_from_list(self, documents: List[Dict[str, str]]):
        """Add documents to the knowledge base."""
        self.vector_store.add_documents(documents)

    def clear(self, delete_persisted_index: bool = False):
        """Clear the vector store and optionally delete the persisted FAISS index."""
        self.vector_store.delete_all()
        self.knowledge_base_loaded = False
        self._query_embed_cache.clear()
        if delete_persisted_index:
            index_path = self._index_cache_path()
            if os.path.exists(index_path):
                os.remove(index_path)
                _tracer.step("faiss_index_deleted", path=index_path)
