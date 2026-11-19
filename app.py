import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(page_title="Gemini PDF Chat (RAG)", page_icon="✨", layout="wide")
st.title("✨ Chat with Your PDF using Google Gemini")

with st.sidebar:
    st.header("1. Gemini Setup")
    api_key = st.text_input(
        "Gemini API Key",
        value=os.getenv("GOOGLE_API_KEY", ""),
        type="password",
        help="Get a free API key from aistudio.google.com"
    )
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    st.header("2. Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    process_button = st.button("Process PDF", type="primary")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []

def build_vector_store(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if process_button:
    if not api_key:
        st.sidebar.error("Please paste your Gemini API Key first.")
    elif not uploaded_file:
        st.sidebar.error("Please upload a PDF file first.")
    else:
        with st.spinner("Scanning PDF and building Gemini index..."):
            st.session_state.vector_store = build_vector_store(uploaded_file)
            st.session_state.messages = []
            st.sidebar.success(f"Ready! '{uploaded_file.name}' has been indexed.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask anything about your PDF..."):
    if not st.session_state.vector_store:
        st.warning("Upload and process a PDF in the left sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking with Gemini..."):
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})

                prompt = ChatPromptTemplate.from_template("""
                Answer the question based strictly on the provided document context.
                If the answer is not in the context, say "I cannot find that information in the uploaded document."

                Context:
                {context}

                Question: {question}
                """)

                llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)

                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                response = rag_chain.invoke(user_query)
                st.markdown(response)

                with st.expander("View Source Chunks"):
                    for i, doc in enumerate(retriever.invoke(user_query), 1):
                        page_num = doc.metadata.get("page", 0) + 1
                        st.markdown(f"**Chunk {i} (Page {page_num}):**")
                        st.caption(doc.page_content)

        st.session_state.messages.append({"role": "assistant", "content": response})