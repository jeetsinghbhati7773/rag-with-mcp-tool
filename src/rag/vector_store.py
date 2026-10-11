from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from src.rag.embeddings import LocalSentenceTransformerEmbeddings
from src.config import settings

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    Persistent ChromaDB vector database manager for document RAG chunks.
    Uses SentenceTransformers 'all-MiniLM-L6-v2' (384-dimensional dense vectors).
    Includes SHA-256 duplicate checking, document lifecycle management, and metadata retrieval.
    """

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        collection_name: str = "knowledge_base",
    ):
        self.persist_directory = persist_directory or settings.chroma_dir
        self.collection_name = collection_name
        self._fallback_memory: List[Document] = []
        if str(self.persist_directory) not in (":memory:", "memory"):
            self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.embedding_function = LocalSentenceTransformerEmbeddings()
        self._init_store()

    def _init_store(self):
        import chromadb
        try:
            if str(self.persist_directory) in (":memory:", "memory"):
                self.client = chromadb.EphemeralClient()
            else:
                self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        except Exception as e:
            logger.warning("Could not initialize Chroma client: %s", e)
            self.client = None

    def _get_collection(self):
        if self.client is None:
            return None
        try:
            return self.client.get_or_create_collection(name=self.collection_name, embedding_function=None)
        except Exception as e:
            logger.warning("Could not get collection: %s", e)
            return None

    def add_documents(self, documents: List[Document], conversation_id: Optional[str] = None) -> int:
        """Ingests semantic document chunks into persistent ChromaDB storage tagged with conversation_id."""
        if not documents:
            return 0

        if conversation_id:
            for doc in documents:
                doc.metadata["conversation_id"] = conversation_id

        texts = [doc.page_content for doc in documents]
        ids = [doc.metadata.get("chunk_id") for doc in documents]

        clean_metadatas = []
        for doc in documents:
            m = {}
            for k, v in (doc.metadata or {}).items():
                if v is not None and isinstance(v, (str, int, float, bool)):
                    m[k] = v
                elif v is not None:
                    m[k] = str(v)
            clean_metadatas.append(m)

        for doc in documents:
            self._fallback_memory.append(doc)

        try:
            col = self._get_collection()
            if col is not None:
                embeddings = self.embedding_function.embed_documents(texts)
                col.add(documents=texts, embeddings=embeddings, metadatas=clean_metadatas, ids=ids)
        except Exception as e:
            logger.warning("ChromaDB add_documents warning: %s; stored in fallback memory", e)

        return len(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        conversation_id: Optional[str] = None,
        document_names: Optional[List[str]] = None,
    ) -> List[Document]:
        """Retrieves top-k most similar document chunks filtered by conversation_id and/or document_names."""
        # If document_names explicitly provided and empty, no documents should match
        if document_names is not None and len(document_names) == 0:
            return []

        try:
            col = self._get_collection()
            if col is not None:
                query_embedding = self.embedding_function.embed_query(query)
                query_kwargs: Dict[str, Any] = {
                    "query_embeddings": [query_embedding],
                    "n_results": k,
                    "include": ["documents", "metadatas", "distances"],
                }

                where_clauses = []
                if conversation_id:
                    where_clauses.append({"conversation_id": conversation_id})
                if document_names:
                    if len(document_names) == 1:
                        where_clauses.append({"document_name": document_names[0]})
                    else:
                        where_clauses.append({"document_name": {"$in": document_names}})

                if len(where_clauses) == 1:
                    query_kwargs["where"] = where_clauses[0]
                elif len(where_clauses) > 1:
                    query_kwargs["where"] = {"$and": where_clauses}

                results = col.query(**query_kwargs)

                retrieved_docs = []
                if results and results.get("documents") and results["documents"][0]:
                    docs_list = results["documents"][0]
                    meta_list = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs_list)
                    dist_list = results["distances"][0] if results.get("distances") else [0.0] * len(docs_list)

                    for doc_text, meta, dist in zip(docs_list, meta_list, dist_list):
                        m = dict(meta or {})
                        m["_distance"] = dist
                        retrieved_docs.append(Document(page_content=doc_text, metadata=m))

                if retrieved_docs:
                    return retrieved_docs
        except Exception as e:
            logger.warning("ChromaDB similarity search warning: %s", e)

        if self._fallback_memory:
            q_lower = query.lower()
            memory = self._fallback_memory
            if document_names is not None:
                memory = [d for d in memory if d.metadata.get("document_name") in document_names]
            elif conversation_id:
                memory = [d for d in memory if d.metadata.get("conversation_id") == conversation_id]
            matches = [d for d in memory if any(w in d.page_content.lower() for w in q_lower.split())]
            return matches[:k] if matches else memory[:k]

        return []

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        conversation_id: Optional[str] = None,
        document_names: Optional[List[str]] = None,
    ) -> List[Tuple[Document, float]]:
        """Retrieves top-k most similar document chunks with distance scores."""
        docs = self.similarity_search(query, k=k, conversation_id=conversation_id, document_names=document_names)
        results = []
        for doc in docs:
            dist = doc.metadata.get("_distance", 0.5)
            results.append((doc, dist))
        return results

    def has_document_hash(self, file_hash: str) -> bool:
        """Checks if a document with given SHA-256 hash exists in vector DB."""
        if not file_hash:
            return False
        try:
            col = self._get_collection()
            if col is not None:
                res = col.get(where={"file_hash": file_hash}, limit=1)
                if res and res.get("ids") and len(res["ids"]) > 0:
                    return True
        except Exception as e:
            logger.debug("Error checking hash in ChromaDB: %s", e)

        return any(d.metadata.get("file_hash") == file_hash for d in self._fallback_memory)

    def get_document_by_hash(self, file_hash: str) -> Dict[str, Any]:
        """Gets document metadata dict for given file hash."""
        try:
            col = self._get_collection()
            if col is not None:
                res = col.get(where={"file_hash": file_hash}, limit=1, include=["metadatas"])
                if res and res.get("metadatas") and len(res["metadatas"]) > 0:
                    return res["metadatas"][0]
        except Exception:
            pass
        for d in self._fallback_memory:
            if d.metadata.get("file_hash") == file_hash:
                return d.metadata
        return {}

    def delete_document(self, document_name: str) -> int:
        """Deletes all chunks belonging to a document by document_name."""
        deleted_count = 0
        try:
            col = self._get_collection()
            if col is not None:
                res = col.get(where={"document_name": document_name}, include=["ids"])
                ids = res.get("ids") or []
                if ids:
                    col.delete(ids=ids)
                    deleted_count = len(ids)
        except Exception as e:
            logger.warning("Error deleting document '%s' from ChromaDB: %s", document_name, e)

        # Remove from fallback memory
        orig_len = len(self._fallback_memory)
        self._fallback_memory = [
            d for d in self._fallback_memory
            if d.metadata.get("document_name") != document_name and d.metadata.get("source") != document_name
        ]
        fallback_deleted = orig_len - len(self._fallback_memory)
        return max(deleted_count, fallback_deleted)

    def list_documents(self) -> List[Dict[str, Any]]:
        """Lists indexed documents with full metadata details."""
        doc_map: Dict[str, Dict[str, Any]] = {}
        try:
            col = self._get_collection()
            if col is not None:
                data = col.get(include=["metadatas"])
                metadatas = data.get("metadatas") or []
                for meta in metadatas:
                    if isinstance(meta, dict):
                        doc_name = meta.get("document_name") or meta.get("source") or "Unknown"
                        if doc_name not in doc_map:
                            doc_map[doc_name] = {
                                "Document Name": doc_name,
                                "Chunks": 0,
                                "Pages": meta.get("total_pages", "-"),
                                "Hash": meta.get("file_hash", "")[:8] if meta.get("file_hash") else "-",
                                "Embedding Model": meta.get("embedding_model", "all-MiniLM-L6-v2"),
                            }
                        doc_map[doc_name]["Chunks"] += 1
                if doc_map:
                    return list(doc_map.values())
        except Exception as e:
            logger.warning("Could not list ChromaDB documents: %s", e)

        for d in self._fallback_memory:
            name = d.metadata.get("document_name") or d.metadata.get("source") or "Unknown"
            if name not in doc_map:
                doc_map[name] = {
                    "Document Name": name,
                    "Chunks": 0,
                    "Pages": d.metadata.get("total_pages", "-"),
                    "Hash": d.metadata.get("file_hash", "")[:8] if d.metadata.get("file_hash") else "-",
                    "Embedding Model": "all-MiniLM-L6-v2",
                }
            doc_map[name]["Chunks"] += 1

        return list(doc_map.values())

    def count_chunks(self) -> int:
        """Returns total number of chunks indexed in ChromaDB."""
        try:
            col = self._get_collection()
            if col is not None:
                cnt = col.count()
                if cnt > 0:
                    return cnt
        except Exception:
            pass
        return len(self._fallback_memory)

    def clear_documents(self):
        """Deletes all collections and clears vector memory."""
        self._fallback_memory.clear()
        try:
            if self.client:
                self.client.delete_collection(self.collection_name)
        except Exception as e:
            logger.warning("Failed to clear ChromaDB collection: %s", e)
