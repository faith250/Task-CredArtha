import os
import gradio as gr
import traceback
from pathlib import Path

# Import necessary functions
# Uncomment if you have these modules
# from data_preparation import create_sample_documents, create_sample_financial_data
from rag_system import FinancialRAGSystem

# Define paths
# If you already have documents, just point to their location
DOCS_DIR = "data/documents"  # Update with your actual documents directory
MODEL_DIR = "models/fine_tuned/final_model"  # Update with your actual model path

# Ensure directories exist
os.makedirs(DOCS_DIR, exist_ok=True)

# Create sample data if needed
# Uncomment if you have these functions
# print("Creating sample data...")
# sample_file = create_sample_financial_data()
# docs_dir = create_sample_documents()

# Initialize the RAG system with proper error handling
try:
    print(f"Initializing RAG system with documents from {DOCS_DIR}")
    rag_system = FinancialRAGSystem(
        docs_dir=DOCS_DIR,
        model_dir=MODEL_DIR
    )
    system_ready = True
    error_message = None
except Exception as e:
    print(f"Error initializing RAG system: {e}")
    traceback.print_exc()
    system_ready = False
    error_message = str(e)

# Function to answer queries
def answer_query(query):
    if not system_ready:
        return f"System initialization failed: {error_message}. Please check the console logs."
    
    if not query.strip():
        return "Please enter a question."
    
    try:
        return rag_system.answer_question(query)
    except Exception as e:
        traceback.print_exc()
        return f"Error processing query: {str(e)}"

# Create UI
demo = gr.Interface(
    fn=answer_query,
    inputs=gr.Textbox(label="Your Question", placeholder="Ask about credit reports, scores, bureau data..."),
    outputs=gr.Textbox(label="FINA's Response"),
    title="FINA - Financial Intelligence & Analytics Assistant",
    description="A specialized LLM for financial and credit bureau data analysis" + 
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