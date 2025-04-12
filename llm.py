import os
import pandas as pd
from langchain.schema import BaseRetriever
from langchain.schema.document import Document
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.document_compressors import JinaRerank
from langchain.retrievers import ContextualCompressionRetriever
from fuzzywuzzy import fuzz
from typing import List
from dotenv import load_dotenv

def load_environment():
    """Load environment variables from .env file"""
    load_dotenv()

def load_puskesmas_data():
    """Load and preprocess puskesmas data from CSV"""
    df = pd.read_csv("data_semua_puskesmas.csv")
    df['nama_layanan'] = df['nama_layanan'].str.strip().str.lower()
    df['nama_puskesmas'] = df['nama_puskesmas'].str.strip().str.lower()
    return df

def create_alias_map():
    """Create mapping of service names to their aliases"""
    return {
        "UGD": ["unit gawat darurat", "gawat darurat", "darurat", "igd", "emergensi", "emergency", "ugd"],
        "Ruang Bersalin": ["bersalin", "melahirkan", "lahiran", "ruang bersalin"],
        "Rawat Jalan": ["rawat jalan", "poli umum", "pelayanan umum", "pemeriksaan umum"],
        "Pelayanan Gigi dan Mulut": ["gigi", "mulut", "dokter gigi", "periksa gigi", "poli gigi"],
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

def prepare_documents(df):
    """Create document objects from dataframe rows"""
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

def setup_embeddings():
    """Initialize embeddings model"""
    return HuggingFaceInferenceAPIEmbeddings(
        api_key=os.getenv("HUGGINGFACE_API_KEY"), 
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

def setup_vector_store(documents, embeddings):
    """Set up Chroma vector store with documents"""
    chroma_dir = os.getenv("CHROMA_DIR")
    
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
        db = Chroma(chroma_dir, embedding_function=embeddings)
    
    return db

def setup_retriever(db):
    """Configure the base retriever with reranking"""
    retriever = db.as_retriever(search_kwargs={'k': 10})
    
    compressor = JinaRerank(
        jina_api_key=os.getenv("JINA_API_KEY"),
        top_n=5,
        model="jina-reranker-v2-base-multilingual",
    )

    return ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

def is_jam_layanan_query(query: str) -> bool:
    """Detect if query is about service hours"""
    keywords = ["jam", "jadwal", "hari", "buka", "tutup", "operasional"]
    return any(k in query.lower() for k in keywords)

def detect_query_type(query: str) -> str:
    """Identify the type of information being requested"""
    query = query.strip().lower()

    if any(word in query for word in ['jam', 'jadwal', 'hari', 'buka', 'tutup', 'operasional']):
        return "jam"
    elif any(word in query for word in ['telepon', 'telpon', 'hp', 'nomor', 'kontak', 'nomor telpon']):
        return "telepon"
    elif "whatsapp" in query or "wa" in query:
        return "whatsapp"
    elif "layanan" in query or "ada layanan apa" in query or "punya layanan apa":
        return "nama_layanan"
    else:
        return "umum"

class PuskesmasKeywordRetriever(BaseRetriever):
    """Custom retriever for puskesmas information with keyword matching"""
    
    def __init__(self, df, layanan_list, alias_map, fallback_retriever):
        super().__init__()
        self._df = df
        self._layanan_list = layanan_list
        self._alias_map = alias_map
        self._fallback_retriever = fallback_retriever
        self._puskesmas_list = df['nama_puskesmas'].unique().tolist()

    def get_layanan_list_from_query(self, query: str):
        """Extract service types from query using direct and fuzzy matching"""
        query = query.lower()
        layanan_terdeteksi = []

        # Alias direct match
        for layanan, aliases in self._alias_map.items():
            if any(alias in query for alias in aliases) and layanan not in layanan_terdeteksi:
                layanan_terdeteksi.append(layanan)

        # Fuzzy matching
        for layanan in self._layanan_list:
            score = fuzz.partial_ratio(layanan.lower(), query)
            if score > 80 and layanan not in layanan_terdeteksi:
                layanan_terdeteksi.append(layanan)

        return layanan_terdeteksi

    def get_filter_from_query(self, query: str):
        """Extract service and puskesmas information from query"""
        layanan_list = self.get_layanan_list_from_query(query)
        matched_puskesmas = None

        for pusk in self._puskesmas_list:
            if fuzz.partial_ratio(pusk.lower(), query.lower()) > 85:
                matched_puskesmas = pusk
                break

        return layanan_list, matched_puskesmas

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents based on query analysis"""
        layanan_list, puskesmas = self.get_filter_from_query(query)
        is_jam_query = is_jam_layanan_query(query)

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

def setup_llm():
    """Initialize the language model"""
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=512,
        timeout=None,
        api_key=os.getenv("GROQ_API_KEY"),
    )

def setup_qa_chain(llm, retriever):
    """Create the question answering chain"""
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

def main():
    """Main function to run the puskesmas information system"""
    # Initialize environment and data
    load_environment()
    df = load_puskesmas_data()
    alias_map = create_alias_map()
    
    # Extract lists for easier access
    layanan_list = df['nama_layanan'].tolist()
    
    # Setup embeddings and vector store
    documents = prepare_documents(df)
    embeddings = setup_embeddings()
    db = setup_vector_store(documents, embeddings)
    
    # Setup retrievers
    compression_retriever = setup_retriever(db)
    custom_retriever = PuskesmasKeywordRetriever(
        df=df,
        layanan_list=layanan_list,
        alias_map=alias_map,
        fallback_retriever=compression_retriever,
    )
    
    # Setup LLM and QA chain
    llm = setup_llm()
    qa_chain = setup_qa_chain(llm, custom_retriever)
    
    # Run chat loop
    print("Sistem Informasi Puskesmas siap digunakan! Ketik 'exit' atau 'quit' untuk keluar.")
    while True:
        user_input = input("🗣️ Pertanyaan: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        result = qa_chain.run(user_input)
        print("🤖 Jawaban:", result)

if __name__ == "__main__":
    main()