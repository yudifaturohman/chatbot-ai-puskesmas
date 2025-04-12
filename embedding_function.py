import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

load_dotenv()

def get_embedding_function():
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.getenv('HUGGINGFACE_API_KEY'), model_name="sentence-transformers/all-mpnet-base-v2"
    )
    return embeddings
