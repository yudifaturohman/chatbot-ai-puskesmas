import os
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain.schema import BaseRetriever
from langchain.schema.document import Document
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_compressors import JinaRerank
from langchain.retrievers import ContextualCompressionRetriever
from fuzzywuzzy import fuzz
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory

def load_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess data from CSV file"""
    df = pd.read_csv(file_path)
    df['nama_layanan'] = df['nama_layanan'].str.strip().str.lower()
    df['nama_puskesmas'] = df['nama_puskesmas'].str.strip().str.lower()
    return df

def create_alias_map() -> Dict[str, List[str]]:
    """Create a mapping of service aliases"""
    return {
        "UGD": ["unit gawat darurat", "gawat darurat", "darurat", "igd", "emergensi", "emergency", "ugd"],
        "Ruang Bersalin": ["bersalin", "melahirkan", "lahiran", "ruang bersalin"],
        "Rawat Jalan": ["rawat jalan", "poli umum", "pelayanan umum", "pemeriksaan umum"],
        "Pelayanan Gigi dan Mulut": ["gigi", "mulut", "dokter gigi", "periksa gigi", "poli gigi"],
        "Pelayanan Gigi & Mulut": ["gigi", "mulut", "dokter gigi", "periksa gigi", "poli gigi"],
        "Pelayanan Kesehatan Jiwa": ["jiwa", "mental", "kejiwaan", "gangguan jiwa", "psikolog"],
        "Pelayanan Kesehatan Ibu & Anak (KIA)": ["kia", "ibu dan anak", "kesehatan ibu", "bayi", "balita", "imunisasi", "posyandu"],
        "Pelayanan KB": ["kb", "keluarga berencana", "kontrasepsi", "suntik kb", "pil kb", "spiral"],
        "Persalinan": ["persalinan", "proses melahirkan", "melahirkan", "lahiran"],
        "Pelayanan Balita, Anak & Remaja": ["balita", "anak", "remaja", "anak-anak", "tumbuh kembang"],
        "Klaster 3 Dewasa dan Lansia": ["dewasa", "lansia", "manula", "orang tua", "pemeriksaan lansia"],
        "Klaster 2 Ibu Hamil dan Nifas": ["ibu hamil", "nifas", "pemeriksaan kehamilan", "pasca melahirkan"],
        "TB Paru": ["tb paru", "tbc", "tb", "tuberkulosis", "paru-paru", "batuk lama"],
        "IMS & Kusta": ["ims", "infeksi menular seksual", "kusta", "lepra", "penyakit kulit menular"],
        "HIV & IMS": ["hiv", "aids", "ims", "penyakit menular seksual"],
        "Pelayanan Poli TB - Paru": ["poli tb", "tb paru", "tbc", "poli paru-paru"],
        "UGD & Persalinan": ["ugd dan persalinan", "emergency lahiran", "darurat melahirkan"],
        "Pelayanan Umum": ["umum", "periksa umum", "rawat jalan", "pemeriksaan biasa"],
    }

def create_documents_from_df(df: pd.DataFrame) -> List[Document]:
    """Create document objects from dataframe"""
    return [
        Document(
            page_content=(
                f"{row['nama_puskesmas']} menyediakan layanan {row['nama_layanan']} "
                f"pada hari {row['hari']} jam {row['jam']}. "
                f"Telepon: {row['telepon']}, WhatsApp: {row['whatsapp']}."
            )
        )
        for _, row in df.iterrows()
    ]

def initialize_embeddings(api_key: str) -> HuggingFaceInferenceAPIEmbeddings:
    """Initialize embeddings model"""
    return HuggingFaceInferenceAPIEmbeddings(
        api_key=api_key, 
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

def setup_compression_retriever(retriever, jina_api_key: str) -> ContextualCompressionRetriever:
    """Set up compression retriever with reranking"""
    compressor = JinaRerank(
        jina_api_key=jina_api_key,
        top_n=5,
        model="jina-reranker-v2-base-multilingual",
    )
    return ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=retriever
    )

class PuskesmasKeywordRetriever(BaseRetriever):
    """Custom retriever for keyword-based retrieval of Puskesmas information"""
    
    def __init__(self, df: pd.DataFrame, alias_map: Dict[str, List[str]], fallback_retriever):
        super().__init__()
        self._df = df
        self._layanan_list = df['nama_layanan'].unique().tolist()
        self._alias_map = alias_map
        self._fallback_retriever = fallback_retriever
        self._puskesmas_list = df['nama_puskesmas'].unique().tolist()

    def get_layanan_list_from_query(self, query: str) -> List[str]:
        """Extract service names from query using direct and fuzzy matching"""
        query = query.lower()
        layanan_terdeteksi = []

        # Alias direct match
        for alias, synonyms in self._alias_map.items():
            for synonym in synonyms:
                if synonym in query and alias.lower() not in layanan_terdeteksi:
                    layanan_terdeteksi.append(alias.lower())

        # Fuzzy matching
        for layanan in self._layanan_list:
            score = fuzz.partial_ratio(layanan.lower(), query)
            if score > 80 and layanan not in layanan_terdeteksi:
                layanan_terdeteksi.append(layanan)

        return layanan_terdeteksi

    def get_filter_from_query(self, query: str) -> tuple:
        """Extract service and puskesmas filters from query"""
        layanan_list = self.get_layanan_list_from_query(query)
        matched_puskesmas = None

        for pusk in self._puskesmas_list:
            if fuzz.partial_ratio(pusk.lower(), query.lower()) > 85:
                matched_puskesmas = pusk
                break

        return layanan_list, matched_puskesmas

    def is_jam_layanan_query(self, query: str) -> bool:
        """Detect if query is about service hours"""
        keywords = ["jam", "jadwal", "hari", "buka", "tutup", "operasional"]
        return any(k in query.lower() for k in keywords)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents based on the query"""
        layanan_list, puskesmas = self.get_filter_from_query(query)
        is_jam_query = self.is_jam_layanan_query(query)

        df_filtered = self._df
        if puskesmas:
            df_filtered = df_filtered[df_filtered['nama_puskesmas'].str.lower() == puskesmas.lower()]

        if layanan_list:
            df_filtered = df_filtered[df_filtered['nama_layanan'].isin(layanan_list)]

        if not df_filtered.empty:
            documents = []
            for _, row in df_filtered.iterrows():
                if is_jam_query:
                    content = (
                        f"Layanan {row['nama_layanan']} di {row['nama_puskesmas']} buka pada hari {row['hari']} "
                        f"jam {row['jam']}."
                    )
                else:
                    content = (
                        f"{row['nama_puskesmas']} menyediakan layanan {row['nama_layanan']} "
                        f"pada hari {row['hari']} jam {row['jam']}. "
                        f"Telepon: {row['telepon']}, WhatsApp: {row['whatsapp']}."
                    )
                documents.append(Document(page_content=content))
            return documents

        return self._fallback_retriever.get_relevant_documents(query)

def initialize_llm(groq_api_key: str) -> ChatGroq:
    """Initialize the LLM model"""
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=512,
        timeout=None,
        api_key=groq_api_key,
    )

def create_contextualize_prompt() -> ChatPromptTemplate:
    """Create prompt for contextualizing questions"""
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "formulate a standalone question that can be understood without the chat history. "
        "You must not introduce any new information or make assumptions beyond what the user already said. "
        "Only rewrite the question using the existing words and references. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

def get_current_time():
    now = datetime.now(ZoneInfo("Asia/Jakarta")).hour

    if 5 <= now < 12:
        return "Selamat pagi!"
    elif 12 <= now < 15:
        return "Selamat siang!"
    elif 15 <= now < 18:
        return "Selamat sore!"
    else:
        return "Selamat malam!"

def create_qa_prompt() -> ChatPromptTemplate:
    """Create prompt for QA chain"""
    prompt_template = """
        Kamu adalah asisten layanan masyarakat puskesmas yang ramah dan informatif. 
        Jawablah pertanyaan dari warga dengan jelas, mudah dipahami, dan dalam gaya bahasa yang sederhana.

        Berikan sapaan pertama kali yang sesuai dengan Waktu saat ini {current_time}.

        Gunakan hanya informasi yang relevan yang diberikan dalam bagian "Informasi yang relevan".
        Jika layanan tidak ditemukan, jawab dengan jujur dan arahkan agar mereka bisa menanyakan ulang.

        Jika pertanyaannya berkaitan dengan jam layanan, sebutkan hari dan jam bukanya berdasarkan informasi yang tersedia.
        Jangan pernah membuat jawaban berdasarkan asumsi atau informasi yang tidak ada dalam konteks.

        Pertanyaan: 
        {input}

        Informasi yang relevan:
        {context}

        Jawaban:
        """
    
    prompt_template = prompt_template.format(
        current_time=get_current_time(),
        input="{input}",
        context="{context}"
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

def setup_rag_chain(llm, retriever):
    """Set up the RAG chain with conversation history"""
    contextualize_q_prompt = create_contextualize_prompt()
    qa_prompt = create_qa_prompt()

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

def get_session_history(session_id: str, connection_string: str) -> BaseChatMessageHistory:
    """Get chat history for a session"""
    return PostgresChatMessageHistory(
        session_id=session_id,
        connection_string=connection_string,
        table_name="message_store",
    )

def setup_conversational_chain(rag_chain, postgres_connection_string: str):
    """Set up conversational chain with history"""
    def get_history_func(session_id: str) -> BaseChatMessageHistory:
        return get_session_history(session_id, postgres_connection_string)

    return RunnableWithMessageHistory(
        rag_chain,
        get_history_func,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )