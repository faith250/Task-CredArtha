import os
import gradio as gr
from pathlib import Path
import traceback
from typing import List

# Import LangChain components
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

# Define your model paths and document directory
MODEL_PATH = r"C:\Users\PC\OneDrive\Cre\Task_Credartha\task_2\fina_project\fina_llm\models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
DOCS_DIR = "data/documents"  # Update with your actual documents directory

# Ensure directories exist
os.makedirs(DOCS_DIR, exist_ok=True)

# Initialize system state variables
system_ready = False
error_message = "System not yet initialized"
rag_chain = None
vectorstore = None

def load_documents():
    """Load and index documents for retrieval"""
    global vectorstore
    
    try:
        print(f"Loading documents from {DOCS_DIR}")
        
        # Check if directory exists and has files
        if not os.path.exists(DOCS_DIR):
            print(f"Creating documents directory: {DOCS_DIR}")
            os.makedirs(DOCS_DIR, exist_ok=True)
        
        # Count files in directory
        doc_files = list(Path(DOCS_DIR).glob("**/*.txt"))
        print(f"Found {len(doc_files)} document files")
        
        if len(doc_files) == 0:
            print("No documents found. Creating sample documents...")
            # Import and create sample documents if needed
            try:
                from data_preparation import create_sample_documents
                create_sample_documents(DOCS_DIR)
                print("Sample documents created successfully")
            except ImportError:
                print("Could not import data_preparation module to create sample documents")
        
        # Load documents
        loader = DirectoryLoader(
            DOCS_DIR,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")
        
        if len(documents) == 0:
            return False, "No documents found or loaded"
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        
        # Create vector store
        print("Initializing embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        print("Creating vector store...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        print("Vector store created successfully")
        
        return True, None
    except Exception as e:
        traceback.print_exc()
        return False, f"Error loading documents: {str(e)}"

def initialize_llm():
    """Initialize the LLM with LangChain"""
    try:
        print(f"Initializing LLM from {MODEL_PATH}")
        
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            return False, f"Model not found at: {MODEL_PATH}"
        
        # Initialize LlamaCpp model for LangChain
        llm = LlamaCpp(
            model_path=MODEL_PATH,
            temperature=0.7,
            max_tokens=512,
            top_p=0.9,
            n_ctx=2048,
            verbose=False  # Set to True for debugging
        )
        
        print("LLM initialized successfully")
        return True, llm
    except Exception as e:
        traceback.print_exc()
        return False, f"Error initializing LLM: {str(e)}"

def setup_rag_chain(llm):
    """Set up the RAG chain with LangChain"""
    global rag_chain, vectorstore
    
    try:
        print("Setting up RAG chain...")
        
        # Create the prompt template
        template = """<|im_start|>system
You are FINA, a Financial Intelligence & Analytics Assistant specializing in credit bureau data and financial analysis.
Use the following context to answer the question. If you don't know the answer based on the context, say that you don't know.

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
"""
        
        prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=template
        )
        
        # Set up memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create the RAG chain
        rag_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            verbose=True
        )
        
        print("RAG chain set up successfully")
        return True, None
    except Exception as e:
        traceback.print_exc()
        return False, f"Error setting up RAG chain: {str(e)}"

def initialize_system():
    """Initialize the complete system"""
    global system_ready, error_message, rag_chain
    
    try:
        # Step 1: Load and index documents
        success, error = load_documents()
        if not success:
            system_ready = False
            error_message = error
            return False
        
        # Step 2: Initialize LLM
        success, llm_or_error = initialize_llm()
        if not success:
            system_ready = False
            error_message = llm_or_error
            return False
        
        # Step 3: Set up RAG chain
        success, error = setup_rag_chain(llm_or_error)
        if not success:
            system_ready = False
            error_message = error
            return False
        
        # All steps succeeded
        system_ready = True
        error_message = None
        return True
    except Exception as e:
        traceback.print_exc()
        system_ready = False
        error_message = f"Error initializing system: {str(e)}"
        return False

# Try to initialize the system
print("Initializing system...")
initialize_system()
print(f"System ready: {system_ready}")
if not system_ready:
    print(f"Error: {error_message}")

def answer_query(query):
    """Process a query through the RAG system"""
    if not system_ready:
        return f"System initialization failed: {error_message}. Please check the console logs."
    
    if not query.strip():
        return "Please enter a question."
    
    try:
        # Process the query through the RAG chain
        result = rag_chain({"question": query})
        
        # Extract and format the answer
        answer = result.get("answer", "No answer generated")
        
        # Get source documents for reference
        source_docs = result.get("source_documents", [])
        
        # Format the response
        if source_docs:
            source_info = "\n\nReferences:"
            for i, doc in enumerate(source_docs[:3], 1):  # Limit to top 3 sources
                source_name = getattr(doc.metadata, 'source', f"Source {i}")
                source_info += f"\n{i}. {os.path.basename(source_name)}"
            response = f"{answer}{source_info}"
        else:
            response = answer
            
        return response
    except Exception as e:
        traceback.print_exc()
        return f"Error processing query: {str(e)}"

# Create UI
demo = gr.Interface(
    fn=answer_query,
    inputs=gr.Textbox(label="Your Question", placeholder="Ask about credit reports, scores, bureau data..."),
    outputs=gr.Textbox(label="FINA's Response"),
    title="FINA - Financial Intelligence & Analytics Assistant",
    description="A specialized LLM for financial and credit bureau data analysis with LangChain RAG" + 
               ("" if system_ready else " (SYSTEM OFFLINE - See console for errors)"),
    examples=[
        ["What is a credit score?"],
        ["What are the main factors that affect my credit score?"],
        ["How is credit utilization calculated?"],
        ["What does a '30 days past due' status mean on my credit report?"],
        ["How long do negative items stay on my credit report?"]
    ]
)

# Start the application
if __name__ == "__main__":
    print("Launching Gradio interface...")
    demo.launch(share=True)  # Set share=False if you don't want a public link