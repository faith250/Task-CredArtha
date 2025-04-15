import os
import gradio as gr
from data_preparation import create_sample_documents, create_sample_financial_data
from rag_system import FinancialRAGSystem, mock_fine_tuning

# Prepare data
print("Creating sample data...")
sample_file = create_sample_financial_data()
docs_dir = create_sample_documents()

# Simulate fine-tuning process
model_dir = mock_fine_tuning()

# Initialize the RAG system
rag_system = FinancialRAGSystem(docs_dir)

# Create Gradio interface
def answer_query(query):
    try:
        return rag_system.answer_question(query)
    except Exception as e:
        return f"Error processing query: {str(e)}"

# Create UI
demo = gr.Interface(
    fn=answer_query,
    inputs=gr.Textbox(label="Your Question", placeholder="Ask about credit reports, scores, bureau data..."),
    outputs=gr.Textbox(label="FINA's Response"),
    title="FINA - Financial Intelligence & Analytics Assistant",
    description="A specialized LLM for financial and credit bureau data analysis",
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
    demo.launch()