import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

# 🔠 Initialize embedding + LLM
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=groq_api_key,  # Pass API key here
    model="openai/gpt-oss-20b",
    temperature=0.7,
    max_tokens=2000,  # Use max_tokens here
)
print(f"Using LLM: {llm.__class__.__name__}")


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
        query = st.text_input(
            "🔎 **Ask your question**",
            placeholder="e.g. What are the symptoms of diabetes?",
        )
        submit = st.form_submit_button("Get Answer 💬")

        if submit and query:  # ✅ Move inside form block
            with st.spinner("Thinking... 🤔"):
                results = vectorstore.similarity_search(query, k=3)
                context = "\n\n".join([doc.page_content for doc in results])

                messages = [
                    SystemMessage(
                        content="You are a helpful assistant answering only from the provided context."
                    ),
                    HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{query}"),
                ]

                response = llm.invoke(messages)

                st.markdown("### 💡 **Answer**")
                st.success(response.content)

                with st.expander("📚 **See Top Matching Chunks**"):
                    for i, doc in enumerate(results):
                        st.markdown(f"**Chunk #{i+1}:**\n{doc.page_content}")
    #fiadufiuahf
