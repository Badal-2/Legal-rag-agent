from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    source_name: str = "document"
) -> Dict:
    try:
        # Validate input
        if not text or len(text.strip()) == 0:
            return {
                "success": False,
                "error": "Text is empty or None"
            }
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Split priority
        )
        
        # Split the text
        chunks = text_splitter.split_text(text)
        
        # Create chunk objects with metadata
        chunk_objects = []
        for i, chunk in enumerate(chunks):
            chunk_obj = {
                "id": f"{source_name}_chunk_{i+1}",
                "text": chunk,
                "chunk_index": i + 1,
                "total_chunks": len(chunks),
                "source": source_name,
                "char_count": len(chunk),
                "word_count": len(chunk.split())
            }
            chunk_objects.append(chunk_obj)
        
        # Calculate statistics
        stats = {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(len(c) for c in chunks) // len(chunks) if chunks else 0,
            "total_characters": len(text),
            "total_words": len(text.split())
        }
        
        return {
            "success": True,
            "chunks": chunk_objects,
            "statistics": stats
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error chunking text: {str(e)}"
        }


def chunk_by_pages(
    page_texts: Dict[int, str],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Dict:
    try:
        if not page_texts:
            return {
                "success": False,
                "error": "No page texts provided"
            }
        
        all_chunks = []
        chunk_counter = 1
        
        # Process each page
        for page_num, text in page_texts.items():
            if not text or len(text.strip()) == 0:
                continue
            
            # Chunk this page's text
            result = chunk_text(
                text=text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                source_name=f"page_{page_num}"
            )
            
            if result["success"]:
                # Add page number to each chunk
                for chunk in result["chunks"]:
                    chunk["page_number"] = page_num
                    chunk["global_chunk_id"] = f"chunk_{chunk_counter}"
                    chunk_counter += 1
                    all_chunks.append(chunk)
        
        return {
            "success": True,
            "chunks": all_chunks,
            "total_pages": len(page_texts),
            "total_chunks": len(all_chunks)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error chunking by pages: {str(e)}"
        }


def get_optimal_chunk_size(text: str) -> int:
    text_length = len(text)
    
    if text_length < 5000:
        return 500  # Small document
    elif text_length < 20000:
        return 1000  # Medium document
    elif text_length < 100000:
        return 1500  # Large document
    else:
        return 2000  # Very large document
