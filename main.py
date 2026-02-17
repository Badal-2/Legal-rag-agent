# Handles API requests (upload PDF, ask question)

import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import uvicorn
import shutil

# Import our RAG agent
from agent import LegalRAGAgent, format_answer


app = FastAPI(
    title="Legal Document Analyzer - RAG AI Agent",
    description="AI-powered legal document analysis using RAG",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG agent (singleton)
rag_agent = None

def get_agent():
    """Get or create RAG agent instance."""
    global rag_agent
    if rag_agent is None:
        rag_agent = LegalRAGAgent(collection_name="legal_documents")
    return rag_agent


# ===================================
# REQUEST MODELS
# ===================================

class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the payment terms?",
                "top_k": 3
            }
        }




@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the frontend HTML page"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Frontend not found. Please create index.html</h1>",
            status_code=404
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    agent = get_agent()
    stats = agent.vector_store.get_stats()
    
    return {
        "status": "healthy",
        "message": "Legal Document Analyzer API is running! 🚀",
        "documents_in_db": stats.get("total_documents", 0)
    }


@app.post("/api/upload-document")
async def upload_document(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )
        
        # Create documents directory if it doesn't exist
        os.makedirs("documents", exist_ok=True)
        
        # Save uploaded file
        file_path = f"documents/{file.filename}"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"\n📄 Uploaded: {file.filename}")
        
        # Process document with RAG agent
        agent = get_agent()
        result = agent.process_document(file_path)
        
        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=result["error"]
            )
        
        return {
            "success": True,
            "message": "Document processed successfully! ✅",
            "document_info": result["document_info"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )


@app.post("/api/ask-question")
async def ask_question(request: QuestionRequest):
    try:
        agent = get_agent()
        
        # Check if any documents are loaded
        stats = agent.vector_store.get_stats()
        if stats.get("total_documents", 0) == 0:
            raise HTTPException(
                status_code=400,
                detail="No documents uploaded. Please upload a PDF first."
            )
        
        # Get answer from RAG agent
        result = agent.ask_question(
            question=request.question,
            top_k=request.top_k
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=result["error"]
            )
        
        return {
            "success": True,
            "question": result["question"],
            "answer": result["answer"],
            "sources": result["sources"],
            "num_sources": result["num_sources"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error answering question: {str(e)}"
        )


@app.get("/api/extract-clauses")
async def extract_key_clauses():
    try:
        agent = get_agent()
        
        # Check if any documents are loaded
        stats = agent.vector_store.get_stats()
        if stats.get("total_documents", 0) == 0:
            raise HTTPException(
                status_code=400,
                detail="No documents uploaded. Please upload a PDF first."
            )
        
        result = agent.extract_key_clauses()
        
        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=result["error"]
            )
        
        return {
            "success": True,
            "clauses": result["clauses"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting clauses: {str(e)}"
        )


@app.get("/api/document-summary")
async def get_document_summary():
    """
    Generate a summary of the uploaded document.
    """
    try:
        agent = get_agent()
        
        # Check if any documents are loaded
        stats = agent.vector_store.get_stats()
        if stats.get("total_documents", 0) == 0:
            raise HTTPException(
                status_code=400,
                detail="No documents uploaded. Please upload a PDF first."
            )
        
        result = agent.get_document_summary()
        
        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=result["error"]
            )
        
        return {
            "success": True,
            "summary": result["answer"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating summary: {str(e)}"
        )


@app.delete("/api/clear-database")
async def clear_database():
    """
    Clear all documents from the database.
    """
    try:
        agent = get_agent()
        result = agent.clear_database()
        
        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=result["error"]
            )
        
        return {
            "success": True,
            "message": "Database cleared successfully! 🗑️"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing database: {str(e)}"
        )


# ===================================
# RUN THE SERVER
# ===================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 STARTING LEGAL DOCUMENT ANALYZER API")
    print("=" * 70)
    print("📍 Server: http://localhost:8000")
    print("📄 Frontend: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🔧 Health: http://localhost:8000/health")
    print("=" * 70 + "\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )