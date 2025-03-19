import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from ctransformers import AutoModelForCausalLM
from transformers import pipeline

# Function to load PDF and split it into chunks
def load_and_process_pdf(file_path):
    loader = PyPDFLoader(file_path=file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Set up FAISS index and vector store
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(documents=docs, ids=uuids)

    return vector_store

# Function to retrieve relevant documents
def retrieve_docs(vector_store, query):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    source_docs = retriever.invoke(query)
    
    content = ''
    for doc in source_docs:
        content += doc.page_content
    
    return content

# Function to generate a response using the language model
def generate_response(content, question):
    model = AutoModelForCausalLM.from_pretrained(
        "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        model_type="mistral",
        gpu_layers=0
    )

    prompt = f"""
    Answer the user question based on the context given and not prior knowledge.
    ------------------
    context: {content}
    ------------------
    question: {question}
    """
    
    response = model(prompt, max_new_tokens=200)
    return response

# Function to translate response to Hindi
def translate_to_hindi(text):
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")
    translation = translator(text)
    return translation[0]['translation_text']

# Streamlit UI
def main():
    st.title("Summarize & Translate Bot (STBot): AI-Powered Communication")

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

    if uploaded_file is not None:
        # Save the uploaded file
        with open("uploaded_document.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success("Document uploaded successfully!")

        # Load and process the document
        vector_store = load_and_process_pdf("uploaded_document.pdf")
        st.info("Document processing complete! You can now ask questions.")

        # Text input for the question
        question = st.text_input("Ask a question about the document:")

        if question:
            st.info("Processing your question...")

            # Retrieve relevant documents based on the question
            content = retrieve_docs(vector_store, question)

            # Generate a response based on the retrieved content
            response = generate_response(content, question)

            # Display the answer
            st.subheader("Answer:")
            st.write(response)

            # Button to translate response to Hindi
            if st.button("Translate to Hindi"):
                hindi_translation = translate_to_hindi(response)
                st.subheader("Translated Answer (Hindi):")
                st.write(hindi_translation)

if __name__ == "__main__":
    main()
