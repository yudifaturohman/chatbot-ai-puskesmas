import os
import pandas as pd
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_groq import ChatGroq
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever
from typing import List
from langchain.prompts import PromptTemplate

from fuzzywuzzy import fuzz

load_dotenv()

# === Load CSV ===
df = pd.read_csv("data_semua_puskesmas.csv")

# === Embedding Vectorstore untuk fallback ===
docs = []
for _, row in df.iterrows():
    text = (
        f"{row['nama_puskesmas']} menyediakan layanan {row['nama_layanan']} "
        f"pada hari {row['hari']} jam {row['jam']}. "
        f"Telepon: {row['telepon']}, WhatsApp: {row['whatsapp']}."
    )
    docs.append(Document(page_content=text))

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
split_docs = splitter.split_documents(docs)

embedding = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv("HUGGINGFACE_API_KEY"), model_name="sentence-transformers/all-mpnet-base-v2"
)
vectorstore = Chroma.from_documents(split_docs, embedding)
fallback_retriever = vectorstore.as_retriever()

# === Alias Map ===
alias_map = {
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
    "Puskesmas Pontang": ["pontang", "puskesmas pontang"],
}

def is_jam_layanan_query(query: str) -> bool:
    keywords = ["jam", "jadwal", "hari", "buka", "tutup", "operasional"]
    return any(k in query.lower() for k in keywords)

# === Custom Retriever ===
class PuskesmasKeywordRetriever(BaseRetriever):
    def __init__(self, df, layanan_list, alias_map, fallback_retriever):
        super().__init__()
        self._df = df
        self._layanan_list = layanan_list
        self._alias_map = alias_map
        self._fallback_retriever = fallback_retriever
        self._puskesmas_list = df['nama_puskesmas'].unique().tolist()

    def get_filter_from_query(self, query: str):
        query = query.lower()
        matched_layanan = None
        matched_puskesmas = None

        # Cek alias layanan
        for alias, real_name in self._alias_map.items():
            if alias in query:
                matched_layanan = real_name
                break

        # Cek nama layanan secara langsung
        if not matched_layanan:
            for layanan in self._layanan_list:
                if layanan.lower() in query:
                    matched_layanan = layanan
                    break

        # Fuzzy match nama puskesmas
        for pusk in self._puskesmas_list:
            if fuzz.partial_ratio(pusk.lower(), query) > 85:
                matched_puskesmas = pusk
                break

        return matched_layanan, matched_puskesmas
    
    def get_all_layanan_from_query(self, query: str) -> List[str]:
        query = query.lower()
        matched = set()

        # Alias matching
        for alias, real in self._alias_map.items():
            if alias in query:
                matched.add(real)

        # Langsung match
        for layanan in self._layanan_list:
            if layanan.lower() in query:
                matched.add(layanan)

        # Fuzzy match
        for layanan in self._layanan_list:
            score = fuzz.partial_ratio(layanan.lower(), query)
            if score > 80:
                matched.add(layanan)

        return list(matched)


    def get_relevant_documents(self, query: str) -> List[Document]:
        layanan_list = self.get_all_layanan_from_query(query)
        _, puskesmas = self.get_filter_from_query(query)
        is_jam_query = is_jam_layanan_query(query)

        df_filtered = self._df
        if puskesmas:
            df_filtered = df_filtered[df_filtered['nama_puskesmas'].str.lower() == puskesmas.lower()]

        documents = []

        if layanan_list:
            for layanan in layanan_list:
                df_layanan = df_filtered[df_filtered['nama_layanan'].str.lower() == layanan.lower()]
                for _, row in df_layanan.iterrows():
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
        elif puskesmas:
            # Kalau tidak menyebut layanan tapi sebut nama puskesmas, tampilkan semua layanan di sana
            for _, row in df_filtered.iterrows():
                content = (
                    f"{row['nama_puskesmas']} menyediakan layanan {row['nama_layanan']} "
                    f"pada hari {row['hari']} jam {row['jam']}. "
                    f"Telepon: {row['telepon']}, WhatsApp: {row['whatsapp']}."
                )
                documents.append(Document(page_content=content))

        if documents:
            return documents

        # fallback ke vectorstore
        return self._fallback_retriever.get_relevant_documents(query)




# === Inisialisasi Chatbot ===
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=512,
    timeout=None,
    api_key=os.getenv("GROQ_API_KEY"),
)
layanan_list = df['nama_layanan'].unique().tolist()

custom_retriever = PuskesmasKeywordRetriever(
    df=df,
    layanan_list=layanan_list,
    alias_map=alias_map,
    fallback_retriever=fallback_retriever
)

rag_prompt = PromptTemplate.from_template("""
Anda adalah asisten informasi layanan puskesmas kecamatan di Kabupaten Serang. Jawablah hanya berdasarkan dokumen yang diberikan. 
Jika informasi tidak ditemukan, jawab dengan "Maaf, saya tidak menemukan informasi tersebut."

Pertanyaan: {question}

Dokumen:
{context}

Jawaban:
""")

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=custom_retriever)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=custom_retriever,
    chain_type="stuff",
    chain_type_kwargs={
        "prompt": rag_prompt
    },
    return_source_documents=True
)

# === Jalankan chatbot ===
print("Chatbot Layanan Puskesmas (exit untuk keluar)")
while True:
    query = input("User: ")
    if query.lower() in ["exit", "quit"]:
        break
    response = qa_chain.run(query)
    print("Bot:", response)
