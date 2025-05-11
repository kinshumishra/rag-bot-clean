import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# 🗝️ Set API config
os.environ["OPENROUTER_API_KEY"] = "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"
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

st.markdown("---")
st.markdown("Made with ❤️ by Siddharth Mishra")
pdf_file = st.file_uploader("📄 Upload a PDF", type="pdf")

if pdf_file:
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.read())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embedding_model)

    with st.form("qa_form"):
    query = st.text_input("🔎 **Ask your question**", placeholder="e.g. What are the symptoms of diabetes?")
    submit = st.form_submit_button("Get Answer 💬")

if submit and query:
    with st.spinner("Thinking... 🤔"):
        results = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in results])

        messages = [
            SystemMessage(content="You are a helpful assistant answering only from the provided context."),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{query}")
        ]

        response = llm.invoke(messages)

        st.markdown("### 💡 **Answer**")
        st.success(response.content)

        with st.expander("📚 **See Top Matching Chunks**"):
            for i, doc in enumerate(results):
                st.markdown(f"**Chunk #{i+1}:**\n{doc.page_content}")

