🩺 Medi-Bot 

Medi-bot is an AI-driven Medical Assistant that processes user queries, analyzes uploaded documents (like medical PDFs), and provides intelligent answers using LLMs.
The system integrates LangChain, FAISS, and Groq API to enable retrieval-augmented generation (RAG) and conversational memory.

🚀 Features

📄 Upload PDF documents and query them.

🤖 Conversational AI powered by LLaMA-3 / Mixtral via Groq API.

🧠 Memory-based chat with LangChain.

🔍 FAISS vector search for fast document retrieval.

🔤 Embeddings with Sentence Transformers.

💻 Streamlit UI for easy interaction.

🛠️ Installation

Clone the repository

git clone https://github.com/your-username/ATS-PROJECT.git
cd ATS-PROJECT


Create a virtual environment

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows


Install dependencies

pip install -r requirements.txt


Set up environment variables
Create a .env file in the root folder and add your API key:

GROQ_API_KEY=your_api_key_here

▶️ Usage

Depending on which module you want to run, use:

 Run the Streamlit App
streamlit run medibot.py


📦 Requirements

Already included in requirements.txt:

streamlit
langchain
langchain-community
langchain-groq
faiss-cpu
sentence-transformers
python-dotenv
PyPDF2

📂 Project Structure :
ATS-PROJECT/
│── medibot.py                # Streamlit app
│── connect_memory_with_llm.py # Chat with memory
│── document_processing.py     # PDF parsing & embeddings
│── requirements.txt           # Dependencies
│── .env                       # API keys
│── README.md                  # Documentation
└── data/                      # Sample PDFs 