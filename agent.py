import os
from dotenv import load_dotenv
from typing import Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI


from tools.pdf_extractor import extract_text_from_pdf
from tools.chunking_tool import chunk_text, get_optimal_chunk_size
from tools.vector_store import VectorStore

load_dotenv()


class LegalRAGAgent:
    def __init__(self, collection_name: str = "legal_documents"):
        try:
            # Initialize Gemini LLM
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.3,  # Lower = more focused answers
            )
            
            # Initialize vector store
            self.vector_store = VectorStore(collection_name=collection_name)
            
            # Store current document info
            self.current_document = None
            
            print("✅ RAG Agent initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing RAG Agent: {str(e)}")
            raise
    
    
    def process_document(self, pdf_path: str) -> Dict:
        try:
            print(f"\n📄 Processing document: {pdf_path}")
            
            # Step 1: Extract text from PDF
            print("   1️⃣ Extracting text from PDF...")
            extract_result = extract_text_from_pdf(pdf_path)
            
            if not extract_result["success"]:
                return {
                    "success": False,
                    "error": f"PDF extraction failed: {extract_result['error']}"
                }
            
            text = extract_result["text"]
            metadata = extract_result["metadata"]
            
            print(f"      ✅ Extracted {metadata['num_pages']} pages, {metadata['total_words']} words")
            
            # Step 2: Chunk the text
            print("   2️⃣ Chunking text...")
            optimal_size = get_optimal_chunk_size(text)
            
            chunk_result = chunk_text(
                text=text,
                chunk_size=optimal_size,
                chunk_overlap=200,
                source_name=metadata['filename']
            )
            
            if not chunk_result["success"]:
                return {
                    "success": False,
                    "error": f"Chunking failed: {chunk_result['error']}"
                }
            
            chunks = chunk_result["chunks"]
            print(f"      ✅ Created {len(chunks)} chunks (size: {optimal_size} chars)")
            
            # Step 3: Add to vector store
            print("   3️⃣ Storing in vector database...")
            store_result = self.vector_store.add_chunks(chunks)
            
            if not store_result["success"]:
                return {
                    "success": False,
                    "error": f"Vector storage failed: {store_result['error']}"
                }
            
            print(f"      ✅ Stored {store_result['chunks_added']} chunks in vector DB")
            
            # Save document info
            self.current_document = {
                "filename": metadata['filename'],
                "num_pages": metadata['num_pages'],
                "total_words": metadata['total_words'],
                "num_chunks": len(chunks),
                "chunk_size": optimal_size
            }
            
            return {
                "success": True,
                "message": "Document processed successfully!",
                "document_info": self.current_document
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error processing document: {str(e)}"
            }
    
    
    def ask_question(self, question: str, top_k: int = 3) -> Dict:
        try:
            print(f"\n🔍 Question: {question}")
            
            # Step 1: Search for relevant chunks
            print("   1️⃣ Searching vector database...")
            search_result = self.vector_store.search(question, top_k=top_k)
            
            if not search_result["success"]:
                return {
                    "success": False,
                    "error": search_result["error"]
                }
            
            if search_result["count"] == 0:
                return {
                    "success": False,
                    "error": "No relevant information found in the document."
                }
            
            relevant_chunks = search_result["results"]
            print(f"      ✅ Found {len(relevant_chunks)} relevant chunks")
            
            # Step 2: Build context from chunks
            context = "\n\n".join([
                f"[Source: {chunk['metadata'].get('source', 'unknown')}, "
                f"Page: {chunk['metadata'].get('page_number', 'N/A')}]\n"
                f"{chunk['text']}"
                for chunk in relevant_chunks
            ])
            
            # Step 3: Create prompt for LLM
            prompt = f"""You are a legal document analysis assistant. Answer the question based ONLY on the provided context from the document.

Context from document:
{context}

Question: {question}

Instructions:
1. Answer based ONLY on the context provided above
2. If the answer is not in the context, say "I cannot find this information in the document"
3. Be precise and cite the source/page when possible
4. Keep the answer clear and concise

Answer:"""
            
            # Step 4: Get answer from LLM
            print("   2️⃣ Generating answer with Gemini...")
            
            response = self.llm.invoke(prompt)
            answer = response.content
            
            print("      ✅ Answer generated!")
            
            # Format sources
            sources = [
                {
                    "text": chunk["text"][:200] + "...",
                    "source": chunk["metadata"].get("source", "unknown"),
                    "page": chunk["metadata"].get("page_number", "N/A")
                }
                for chunk in relevant_chunks
            ]
            
            return {
                "success": True,
                "question": question,
                "answer": answer,
                "sources": sources,
                "num_sources": len(sources)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error answering question: {str(e)}"
            }
    
    
    def extract_key_clauses(self) -> Dict:
        """
        Extract key clauses from the document automatically.
        
        Returns:
            Dictionary with key clauses
        """
        try:
            key_topics = [
                "payment terms",
                "termination clause",
                "confidentiality",
                "liability",
                "duration of contract",
                "dispute resolution"
            ]
            
            results = {}
            
            for topic in key_topics:
                result = self.ask_question(f"What does the document say about {topic}?", top_k=2)
                if result["success"]:
                    results[topic] = result["answer"]
            
            return {
                "success": True,
                "clauses": results
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error extracting clauses: {str(e)}"
            }
    
    
    def get_document_summary(self) -> Dict:
        """
        Generate a summary of the document.
        
        Returns:
            Dictionary with summary
        """
        try:
            result = self.ask_question(
                "Provide a brief summary of this document covering the main points and purpose.",
                top_k=5
            )
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error generating summary: {str(e)}"
            }
    
    
    def clear_database(self) -> Dict:
        """
        Clear all documents from the vector database.
        
        Returns:
            Dictionary with status
        """
        try:
            result = self.vector_store.delete_collection()
            self.current_document = None
            
            # Reinitialize vector store
            self.vector_store = VectorStore(collection_name=self.vector_store.collection_name)
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error clearing database: {str(e)}"
            }


# ===================================
# HELPER FUNCTIONS
# ===================================

def format_answer(result: Dict) -> str:
    """
    Format the answer result into readable text.
    """
    if not result["success"]:
        return f"❌ Error: {result['error']}"
    
    output = f"\n💬 Question: {result['question']}\n"
    output += "=" * 70 + "\n"
    output += f"\n✅ Answer:\n{result['answer']}\n"
    output += "\n" + "=" * 70 + "\n"
    output += f"\n📚 Sources ({result['num_sources']} chunks used):\n"
    
    for i, source in enumerate(result["sources"], 1):
        output += f"\n   {i}. Source: {source['source']}, Page: {source['page']}\n"
        output += f"      {source['text']}\n"
    
    return output

