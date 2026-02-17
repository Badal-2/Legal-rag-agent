import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


class VectorStore:
    def __init__(self, collection_name: str = "legal_documents"):
        try:
            # Initialize Gemini embeddings
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            
            # Initialize ChromaDB client (persistent storage)
            self.client = chromadb.PersistentClient(
                path="./chroma_db"  # Local storage directory
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Legal document chunks with embeddings"}
            )
            
            self.collection_name = collection_name
            
            print(f"✅ Vector store initialized: {collection_name}")
            print(f"   Current documents: {self.collection.count()}")
            
        except Exception as e:
            print(f"❌ Error initializing vector store: {str(e)}")
            raise
    
    
    def add_chunks(self, chunks: List[Dict]) -> Dict:
        try:
            if not chunks:
                return {
                    "success": False,
                    "error": "No chunks provided"
                }
            
            # Prepare data for ChromaDB
            documents = []  # The actual text
            metadatas = []  # Metadata for each chunk
            ids = []        # Unique IDs
            
            for chunk in chunks:
                documents.append(chunk["text"])
                
                # Create metadata (exclude 'text' to avoid duplication)
                metadata = {
                    "chunk_index": chunk.get("chunk_index", 0),
                    "source": chunk.get("source", "unknown"),
                    "char_count": chunk.get("char_count", 0),
                    "word_count": chunk.get("word_count", 0),
                    "page_number": chunk.get("page_number", 0)
                }
                metadatas.append(metadata)
                
                # Use chunk ID or generate one
                chunk_id = chunk.get("id") or chunk.get("global_chunk_id") or f"chunk_{len(ids)}"
                ids.append(chunk_id)
            
            # Add to ChromaDB (it will auto-generate embeddings)
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            return {
                "success": True,
                "chunks_added": len(chunks),
                "total_documents": self.collection.count()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error adding chunks: {str(e)}"
            }
    
    
    def search(self, query: str, top_k: int = 3) -> Dict:
        """
        Search for similar chunks using semantic search.
        Args:
            query: User's question or search query
            top_k: Number of top results to return
            
        Returns:
            Dictionary with search results
        """
        try:
            if not query or len(query.strip()) == 0:
                return {
                    "success": False,
                    "error": "Query is empty"
                }
            
            # Check if collection has documents
            if self.collection.count() == 0:
                return {
                    "success": False,
                    "error": "No documents in vector store. Please add documents first."
                }
            
            # Search in ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, self.collection.count())
            )
            
            # Format results
            search_results = []
            
            if results and results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        "text": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "id": results['ids'][0][i],
                        "distance": results['distances'][0][i] if 'distances' in results else None
                    }
                    search_results.append(result)
            
            return {
                "success": True,
                "query": query,
                "results": search_results,
                "count": len(search_results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error searching: {str(e)}"
            }
    
    
    def delete_collection(self) -> Dict:
        """
        Delete the entire collection (clear all data).
        
        Returns:
            Dictionary with success status
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            return {
                "success": True,
                "message": f"Collection '{self.collection_name}' deleted"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error deleting collection: {str(e)}"
            }
    
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with collection stats
        """
        try:
            return {
                "success": True,
                "collection_name": self.collection_name,
                "total_documents": self.collection.count(),
                "storage_path": "./chroma_db"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error getting stats: {str(e)}"
            }



# ===================================
# HELPER FUNCTIONS
# ===================================

def create_vector_store(collection_name: str = "legal_documents") -> VectorStore:
    """
    Create a new vector store instance.
    """
    return VectorStore(collection_name=collection_name)


def format_search_results(results: Dict) -> str:
    """
    Format search results into readable text.
    
    Args:
        results: Search results from vector_store.search()
        
    Returns:
        Formatted string
    """
    if not results["success"]:
        return f"❌ Search failed: {results['error']}"
    
    if results["count"] == 0:
        return "No results found."
    
    output = f"\n🔍 Search Results for: '{results['query']}'\n"
    output += f"Found {results['count']} relevant chunks:\n"
    output += "=" * 70 + "\n"
    
    for i, result in enumerate(results["results"], 1):
        output += f"\n📄 Result {i}:\n"
        output += f"   Source: {result['metadata'].get('source', 'unknown')}\n"
        
        if result['metadata'].get('page_number'):
            output += f"   Page: {result['metadata']['page_number']}\n"
        
        output += f"   Text: {result['text'][:200]}...\n"
        output += "-" * 70 + "\n"
    
    return output


