# rag.py
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Dict
import hashlib

# Initialize once
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Persistent storage (saves to disk)
chroma_client = chromadb.Client(Settings(
    chroma_db_dir="./chroma_db",
    chroma_db_impl="duckdb+parquet",
))

class DocumentStore:
    def __init__(self):
        self.embedder = embedder
        self.client = chroma_client
    
    def create_collection(self, name: str):
        """Create new collection for a document"""
        try:
            collection = self.client.create_collection(name=name)
            return collection
        except:
            # Collection exists
            return self.client.get_collection(name=name)
    
    def add_document(self, collection_name: str, chunks: List[Dict]):
        """Add document chunks with embeddings"""
        collection = self.create_collection(collection_name)
        
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(texts).tolist()
        
        collection.add(
            ids=[c["id"] for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"page": c.get("page", 0), "source": c.get("source", "")} for c in chunks]
        )
        
        return len(chunks)
    
    def query(self, collection_name: str, question: str, n_results: int = 3):
        """Search relevant chunks for question"""
        collection = self.client.get_collection(collection_name)
        
        q_embedding = self.embedder.encode([question]).tolist()
        
        results = collection.query(
            query_embeddings=q_embedding,
            n_results=n_results
        )
        
        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0]
        }

# Global instance
doc_store = DocumentStore()