# Chatbot AI Puskesmas

*Chatbot AI Puskesmas* is an AI-based chatbot system that utilizes Retrieval-Augmented Generation (RAG) technology to help people get health service information from Puskesmas quickly, accurately, and easily understood.

This project uses *LangChain* and *Large Language Model (LLM)* frameworks to answer questions from users based on puskesmas service data in CSV format.

## Main Feature

- *Natural language-based question and answer*
- *Multi-service search support* in one inquiry
- *Automatic hours of operation query detection*
- *Aliases and fuzzy matching* for service names (e.g. "darurat" matches to "UGD")
- *Friendly and easy-to-understand answers for the public*
- *Custom retriever* with fallback to vectorstore (ChromaDB)

## Example Question

- "Apa saja layanan di Puskesmas Anyar?"
- "Jam berapa ruang bersalin dan UGD buka di Bandung?"
- "Layanan gigi dan imunisasi ada hari apa?"
- "Apakah ada layanan darurat?"


## Tech Stack

- Python
- Beutifulshop
- LangChain
- Postgree SQL
- Groq Cloud
- Llama3.1

## Requirements

- Python 3.7 or higher
- The following Python packages (listed in `requirements.txt`)

## Setup

1. Clone the repository and navigate to the project directory.
   ```sh
   git clone https://github.com/yudifaturohman/chatbot-ai-puskesmas.git
   cd chatbot-ai-puskesmas
   ```
2. Create environment project.
   ```sh
   python -m venv venv
   ```
3. Activate environment.
   - MacOS
   ```sh
   source venv/bin/activate
   ```
   - Windows
   ```sh
   venv\Scripts\activate.bat
   ```
   or
   
   ```sh
   venv\Scripts\Activate.ps1
   ```
4. Install the required packages using pip:
   ```sh
   pip install -r requirements.txt
   ```
5. Set the environment variable with your project. You can do this by creating a `.env` file in the project directory with the following content:
   ```
    HUGGINGFACE_API_KEY=your_api_key
    CHROMA_DIR=your_directory_name
    JINA_API_KEY=your_api_key
    GROQ_API_KEY=your_api_key
    POSTGRES_CONNECTION_STRING="postgresql://your_username:your_password@localhost/your_database_name"
   ```

## Running the Application

1. To run Chatbot AI, use the following command:

```sh
uvicorn main:app --reload
```
>Example Send POST request to <code>chat/</code>:
```sh
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "Jam layanan gigi di Puskesmas Bandung?", "session_id": "user123"}'
```

2. Running scrape script

```sh
python scraping.py
```


## Task List
- [x] Scraping service data Puskesmas
- [x] Ranking RAG
- [x] Custom Retriever
- [x] Chat with Prompting
- [x] Chat Memory with Postgree SQL
- [ ] Integration with WhatsApp
- [ ] Tool Agent for Create Ticket
