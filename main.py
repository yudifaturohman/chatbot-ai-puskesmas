from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import os
from dotenv import load_dotenv

# Import our modular components except setup_vector_store
from llm_puskesmas_ai import (
    load_data,
    create_alias_map,
    create_documents_from_df,
    initialize_embeddings,
    PuskesmasKeywordRetriever,
    setup_compression_retriever,
    initialize_llm,
    setup_rag_chain,
    setup_conversational_chain
)

# Import Chroma directly for vector store setup
from langchain.vectorstores import Chroma

# Load environment variables
load_dotenv()
ENV_VARS = {
    "huggingface_api_key": os.getenv("HUGGINGFACE_API_KEY"),
    "jina_api_key": os.getenv("JINA_API_KEY"),
    "groq_api_key": os.getenv("GROQ_API_KEY"),
    "chroma_dir": os.getenv("CHROMA_DIR"),
    "postgres_connection_string": os.getenv("POSTGRES_CONNECTION_STRING"),
}

# FastAPI app
app = FastAPI(title="Puskesmas RAG API")

# Initialize components at startup
@app.on_event("startup")
async def startup_event():
    # Load and prepare data
    global conversational_rag_chain
    
    df = load_data("data_semua_puskesmas.csv")
    alias_map = create_alias_map()
    documents = create_documents_from_df(df)
    
    # Set up embeddings
    embeddings = initialize_embeddings(ENV_VARS["huggingface_api_key"])
    
    # Set up vector store directly (not using setup_vector_store function)
    chroma_dir = ENV_VARS["chroma_dir"]
    if not os.path.exists(chroma_dir):
        os.makedirs(chroma_dir)
        db = Chroma(
            collection_name="layanan_puskesmas",
            persist_directory=chroma_dir,
            embedding_function=embeddings,
        )
        db.add_documents(documents)
        db.persist()
    else:
        db = Chroma(
            collection_name="layanan_puskesmas",
            persist_directory=chroma_dir, 
            embedding_function=embeddings
        )
    
    # Set up retriever
    base_retriever = db.as_retriever(search_kwargs={'k': 10})
    compression_retriever = setup_compression_retriever(
        base_retriever, ENV_VARS["jina_api_key"]
    )
    custom_retriever = PuskesmasKeywordRetriever(df, alias_map, compression_retriever)
    
    # Set up LLM and RAG chain
    llm = initialize_llm(ENV_VARS["groq_api_key"])
    rag_chain = setup_rag_chain(llm, custom_retriever)
    
    # Set up conversational chain
    conversational_rag_chain = setup_conversational_chain(
        rag_chain, ENV_VARS["postgres_connection_string"]
    )

# Request model
class QueryRequest(BaseModel):
    query: str
    session_id: str

# Response model
class QueryResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    """
    Endpoint for chatting with the Puskesmas RAG system.
    
    Parameters:
    - query: The user's question
    - session_id: Unique session identifier for conversation tracking
    
    Returns:
    - answer: The system's response
    """
    try:
        # Invoke the chain with the user's query and session_id
        result = conversational_rag_chain.invoke(
            {"input": request.query}, 
            {'configurable': {'session_id': request.session_id}}
        )
        
        return {"answer": result['answer']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}