"""
RAG Logic Module
================
This module contains the core functions for the Retrieval Augmented Generation system.
It is decoupled from any specific interface (CLI or Web).
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini
def configure_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return False
    genai.configure(api_key=api_key)
    return True

def load_documents(folder_path="documents"):
    """
    Load all text documents from the specified folder.
    """
    documents = []
    doc_folder = Path(folder_path)
    
    # Create folder if it doesn't exist
    doc_folder.mkdir(exist_ok=True)
    
    # Read all .txt files
    for file_path in doc_folder.glob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    documents.append((file_path.name, content))
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
    
    return documents

def split_into_chunks(text, chunk_size=500):
    """
    Split text into smaller chunks for better processing.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1  # +1 for space
        
        if current_size >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def create_embeddings_batch(texts, task_type="retrieval_document"):
    """
    Create embeddings for a list of texts using batch processing.
    """
    if not texts:
        return []
        
    try:
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=texts,
            task_type=task_type
        )
        return result['embedding']
    except Exception as e:
        print(f"Error creating batch embeddings: {e}")
        # Fallback to individual
        embeddings = []
        for text in texts:
            try:
                res = genai.embed_content(
                    model="models/gemini-embedding-001",
                    content=text,
                    task_type=task_type
                )
                embeddings.append(res['embedding'])
            except:
                embeddings.append([0.0] * 768)
        return embeddings

def cosine_similarity(vec1, vec2):
    """
    Calculate similarity between two vectors.
    """
    return sum(a * b for a, b in zip(vec1, vec2))

def find_relevant_chunks(question, chunks_with_embeddings, top_k=3):
    """
    Find the most relevant chunks for a question.
    """
    try:
        res = genai.embed_content(
            model="models/gemini-embedding-001",
            content=question,
            task_type="retrieval_query"
        )
        question_embedding = res['embedding']
    except Exception as e:
        print(f"Error creating question embedding: {e}")
        return []
    
    similarities = []
    for chunk, chunk_embedding in chunks_with_embeddings:
        similarity = cosine_similarity(question_embedding, chunk_embedding)
        similarities.append((chunk, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in similarities[:top_k]]

def generate_answer(question, context_chunks):
    """
    Generate an answer using AI based on relevant context.
    """
    context = "\n\n".join(context_chunks)
    
    prompt = f"""You are a helpful assistant that answers questions based on the provided context.
Based on the following context, please answer the question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
    
    models_to_try = ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-pro']
    
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "429" in str(e):
                continue
            elif "404" in str(e):
                continue
            return f"Error: {e}"
            
    return "All models failed (likely due to quota limits)."

def process_documents_for_rag(documents):
    """
    Full pipeline: documents -> chunks -> embeddings
    """
    all_chunks = []
    for _, content in documents:
        all_chunks.extend(split_into_chunks(content))
    
    embeddings = create_embeddings_batch(all_chunks)
    return list(zip(all_chunks, embeddings))
