# ** WORK FLOW **

USER uploads legal PDF document
    ↓
EXTRACT text from PDF
    ↓
DIVIDE text into CHUNKS (small pieces)
    ↓
CONVERT chunks into VECTOR EMBEDDINGS (numbers)
    ↓
SAVE embeddings in VECTOR DB (ChromaDB/Pinecone)
    ↓
USER asks question: "What are the payment terms?"
    ↓
SEMANTIC SEARCH in vector DB (find relevant chunks)
    ↓
Send relevant chunks + question to LLM (Gemini)
    ↓
LLM generates answer based on document
    ↓
USER gets accurate answer







# 2nd


# PHASE 1 (Document Storage):
PDF → Extract → Chunks → Gemini Embedding → ChromaDB
         (text)                (vectors)    (stores all 3)

# PHASE 2 (Question Answering):
Question → Gemini Embedding → Semantic Search in ChromaDB
             (same model!)          ↓
                           TOP K results retrieved
                           (Original Text + Metadata)
                                   ↓
                    LLM reads text + generates answer
                    (NO vectors sent to LLM!)