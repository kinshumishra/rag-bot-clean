import os
import streamlit as st
st.set_page_config(page_title="🧠 RAG Bot", page_icon="📄", layout="centered")
st.title("🧠 RAG PDF QA Bot")
st.subheader("Ask questions from your uploaded medical document 📄")
st.markdown("---")
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# 🗝️ Set API config
os.environ["OPENROUTER_API_KEY"] = "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"  # 🔁 Replace with your real OpenRouter API key
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# 🔠 Initialize embedding + LLM
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.environ["OPENROUTER_API_KEY"]
)

# 🧠 App UI
st.title("🧠 RAG Chatbot for PDFs")

pdf_file = st.file_uploader("📄 Upload a PDF", type="pdf")

if pdf_file:
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.read())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embedding_model)

    query = st.text_input("❓ Ask a question about the document")

    if query:
        results = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in results])

        messages = [
            SystemMessage(content="You are a helpful assistant answering only from the provided context."),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{query}")
        ]

        response = llm.invoke(messages)

        st.markdown("### 💬 GPT Answer")
        st.write(response.content)

        with st.expander("📚 Top Matching Chunks"):
            for i, doc in enumerate(results):
                st.markdown(f"**Chunk #{i+1}:**\n{doc.page_content}")
st.markdown("---")
st.markdown("Made with ❤️ using LangChain, FAISS, HuggingFace, and Streamlit")
