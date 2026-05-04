import streamlit as st
import rag_logic as rag
import time

# Page Configuration
st.set_page_config(
    page_title="Gemini RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize Gemini
if not rag.configure_gemini():
    st.error("Google API Key not found. Please check your .env file.")
    st.stop()

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None

if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0

# Sidebar
with st.sidebar:
    st.title("Settings & Status")
    st.markdown("---")
    
    if st.button("🔄 Reload Documents"):
        with st.spinner("Processing documents..."):
            docs = rag.load_documents()
            if docs:
                st.session_state.knowledge_base = rag.process_documents_for_rag(docs)
                st.session_state.doc_count = len(docs)
                st.success(f"Loaded {len(docs)} documents!")
            else:
                st.warning("No .txt documents found in 'documents' folder.")
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.info(f"**Status:** {st.session_state.doc_count} Documents Loaded")
    if st.session_state.knowledge_base:
        st.write(f"**Chunks:** {len(st.session_state.knowledge_base)}")

# Main Interface
st.title("🤖 Gemini RAG Assistant")
st.markdown("Ask questions about your documents using the power of Gemini AI.")

# Initial load if not already done
if st.session_state.knowledge_base is None:
    docs = rag.load_documents()
    if docs:
        st.session_state.knowledge_base = rag.process_documents_for_rag(docs)
        st.session_state.doc_count = len(docs)

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 View Source Context"):
                for i, source in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1}:**")
                    st.code(source, wrap_lines=True)

# Chat Input
if prompt := st.chat_input("What would you like to know?"):
    if not st.session_state.knowledge_base:
        st.warning("Please add documents to the 'documents' folder and click 'Reload Documents'.")
    else:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching and thinking..."):
                # Find relevant chunks
                relevant_chunks = rag.find_relevant_chunks(prompt, st.session_state.knowledge_base)
                
                # Generate answer
                answer = rag.generate_answer(prompt, relevant_chunks)
                
                # Display answer
                st.markdown(answer)
                
                # Show sources
                if relevant_chunks:
                    with st.expander("🔍 View Source Context"):
                        for i, chunk in enumerate(relevant_chunks):
                            st.markdown(f"**Source {i+1}:**")
                            st.code(chunk, wrap_lines=True)
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": relevant_chunks
                })
