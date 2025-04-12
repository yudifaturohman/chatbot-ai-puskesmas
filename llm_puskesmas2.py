import os
from dotenv import load_dotenv
import pandas as pd
from langchain.schema import Document
from langchain.schema import BaseRetriever
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA  # ganti dengan ChatGroq jika pakai Groq
from langchain_groq import ChatGroq
from rapidfuzz import fuzz
from typing import List

# === Load ENV ===
load_dotenv()

# === Load CSV ===
df = pd.read_csv("data_semua_puskesmas.csv")
layanan_list = df["nama_layanan"].unique().tolist()

# === Alias Map ===
alias_map = {
    "bersalin": "Ruang Bersalin",
    "gigi": "Pelayanan Gigi dan Mulut",
    "ugd": "UGD",
    "kb": "Klaster 3 Pelayanan KB",
    "jiwa": "Pelayanan Kesehatan Jiwa",
    "ibu hamil": "Klaster 2 Ibu Hamil dan Nifas",
    "remaja": "Klaster 2 Bayi dan Remaja",
    "tb": "Pelayanan TB Paru",
    "rawat jalan": "Pelayanan Rawat Jalan"
}

# === Utility ===
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

    def get_all_layanan_from_query(self, query: str) -> List[str]:
        query = query.lower()
        matched = set()

        for alias, real in self._alias_map.items():
            if alias in query:
                matched.add(real)

        for layanan in self._layanan_list:
            if layanan.lower() in query:
                matched.add(layanan)

        for layanan in self._layanan_list:
            score = fuzz.partial_ratio(layanan.lower(), query)
            if score > 80:
                matched.add(layanan)

        return list(matched)

    def get_filter_from_query(self, query: str):
        matched_puskesmas = None
        for pusk in self._puskesmas_list:
            if fuzz.partial_ratio(pusk.lower(), query.lower()) > 85:
                matched_puskesmas = pusk
                break
        return None, matched_puskesmas

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
                            f"Layanan {row['nama_layanan']} di {row['nama_puskesmas']} buka pada hari {row['hari']} jam {row['jam']}."
                        )
                    else:
                        content = (
                            f"{row['nama_puskesmas']} menyediakan layanan {row['nama_layanan']} pada hari {row['hari']} jam {row['jam']}. "
                            f"Telepon: {row['telepon']}, WhatsApp: {row['whatsapp']}."
                        )
                    documents.append(Document(page_content=content))
        elif puskesmas:
            for _, row in df_filtered.iterrows():
                content = (
                    f"{row['nama_puskesmas']} menyediakan layanan {row['nama_layanan']} pada hari {row['hari']} jam {row['jam']}. "
                    f"Telepon: {row['telepon']}, WhatsApp: {row['whatsapp']}."
                )
                documents.append(Document(page_content=content))

        if documents:
            return documents

        return self._fallback_retriever.get_relevant_documents(query)

# === Dummy fallback vector retriever ===
embedding = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv("HUGGINGFACE_API_KEY"), model_name="sentence-transformers/all-mpnet-base-v2"
)
docs = [Document(page_content=row.to_string()) for _, row in df.iterrows()]
vectorstore = Chroma.from_documents(docs, embedding)
fallback_retriever = vectorstore.as_retriever(search_kwargs={'k': 10})

# === Init Custom Retriever ===
puskesmas_retriever = PuskesmasKeywordRetriever(
    df=df,
    layanan_list=layanan_list,
    alias_map=alias_map,
    fallback_retriever=fallback_retriever
)

# === Prompt Template ===
prompt_template = PromptTemplate.from_template("""
Anda adalah asisten informasi layanan puskesmas. Jawablah hanya berdasarkan dokumen berikut.
Jika informasi tidak ditemukan, jawab: "Maaf, saya tidak menemukan informasi tersebut."

Pertanyaan: {question}

Dokumen:
{context}

Jawaban:
""")

# === RAG QA Chain ===
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=512,
    timeout=None,
    api_key=os.getenv("GROQ_API_KEY"),
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,  # atau ChatGroq(model_name="llama3-8b")
    retriever=puskesmas_retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)

# === Contoh Penggunaan ===
query = "puskesmas baros buka layanan apa aja?"
response = rag_chain.invoke({"query": query})
print(response["result"])
