import os
import gradio as gr
from pathlib import Path
import traceback

# Try to import the llama-cpp-python library (you'll need to install this)
try:
    from llama_cpp import Llama
    llama_cpp_available = True
except ImportError:
    print("llama-cpp-python not found. Please install it with: pip install llama-cpp-python")
    llama_cpp_available = False

# Define your model paths
BASE_MODEL_PATH = r"C:\Users\PC\OneDrive\Cre\Task_Credartha\task_2\fina_project\fina_llm\models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
FINE_TUNED_MODEL_DIR = r"C:\Users\PC\OneDrive\Cre\Task_Credartha\task_2\fina_project\fina_llm\models\fine_tuned\final_model"

# Initialize model
model = None
system_ready = False
error_message = "Model not yet loaded"

def load_model():
    global model, system_ready, error_message
    
    if not llama_cpp_available:
        error_message = "llama-cpp-python library not available. Please install it first."
        return False
    
    try:
        # Check if base model exists
        if not os.path.exists(BASE_MODEL_PATH):
            error_message = f"Base model not found at: {BASE_MODEL_PATH}"
            return False
            
        print(f"Loading model from {BASE_MODEL_PATH}")
        
        # For GGUF models, we use the llama-cpp-python library
        model = Llama(
            model_path=BASE_MODEL_PATH,
            n_ctx=2048,  # Context window size
            n_threads=4   # Number of threads to use
        )
        
        print("Model loaded successfully")
        system_ready = True
        error_message = None
        return True
    except Exception as e:
        traceback.print_exc()
        error_message = f"Error loading model: {str(e)}"
        return False

# Try to load the model
load_model()

def generate_response(query):
    """Generate a response using the model"""
    if not system_ready:
        return f"System initialization failed: {error_message}. Please check the console logs."
    
    if not query.strip():
        return "Please enter a question."
    
    try:
        # Prepare the prompt
        prompt = f"""<|im_start|>system
You are FINA, a Financial Intelligence & Analytics Assistant specializing in credit bureau data and financial analysis.
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
"""
        
        # Generate the response
        output = model(
            prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            echo=False,
            stop=["<|im_end|>", "<|im_start|>"]
        )
        
        # Extract the response
        response = output["choices"][0]["text"].strip()
        return response
    except Exception as e:
        traceback.print_exc()
        return f"Error processing query: {str(e)}"

# Create UI
demo = gr.Interface(
    fn=generate_response,
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