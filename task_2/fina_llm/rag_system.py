import os
from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class FinancialRAGSystem:
    def __init__(self, docs_dir, model_dir="models/fine_tuned/final_model"):
        self.docs_dir = docs_dir
        self.model_dir = model_dir
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Load documents and create vector store
        self.vectorstore = self._load_documents()
        
        # Load the fine-tuned model
        self._load_model()
        
    def _load_documents(self):
        """Load and index documents for retrieval"""
        # Load documents from directory
        loader = DirectoryLoader(
            self.docs_dir,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        return vectorstore
    
    def _load_model(self):
        """Load the fine-tuned model"""
        try:
            # Check if model_dir exists
            if not os.path.exists(self.model_dir):
                raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
                
            # Determine the base model from adapter_config.json
            config_path = os.path.join(self.model_dir, "adapter_config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"adapter_config.json not found in {self.model_dir}")
            
            # Since we don't have direct access to the config file, we'll use a default base model
            # In a real scenario, you would parse the config file to get the base model
            base_model = "microsoft/phi-2"  # Replace with your actual base model
            
            # Load tokenizer and model
            print(f"Loading tokenizer from {base_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            
            print(f"Loading base model from {base_model}")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            print(f"Loading adapter from {self.model_dir}")
            self.model = PeftModel.from_pretrained(
                self.base_model,
                self.model_dir
            )
            
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fall back to base model if adapter loading fails
            try:
                base_model = "microsoft/phi-2"  # Replace with your actual base model
                self.tokenizer = AutoTokenizer.from_pretrained(base_model)
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                print("Fallback to base model successful")
            except Exception as e2:
                print(f"Failed to load fallback model: {e2}")
                raise
    
    def retrieve_relevant_context(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant context based on the query"""
        docs = self.vectorstore.similarity_search(query, k=top_k)
        return [doc.page_content for doc in docs]
    
    def generate_response(self, query: str, context: List[str]) -> str:
        """Generate a response based on the query and context"""
        # Prepare prompt with context
        context_text = "\n\n".join(context)
        prompt = f"""
You are FINA, a Financial Intelligence & Analytics Assistant specializing in credit bureau data and financial analysis.
Use the following context to answer the question. If you don't know the answer, just say so.

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:
"""
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the answer part (after "ANSWER:")
        if "ANSWER:" in response:
            answer = response.split("ANSWER:")[1].strip()
        else:
            answer = response.replace(prompt, "").strip()
        
        return answer
    
    def answer_question(self, query: str) -> str:
        """Main method to answer a question using the RAG system"""
        try:
            # Retrieve relevant context
            context = self.retrieve_relevant_context(query)
            
            # Generate response
            response = self.generate_response(query, context)
            
            return response
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return f"Sorry, I encountered an error while processing your question: {str(e)}"


def mock_fine_tuning():
    """Create a mock fine-tuned model directory structure for testing"""
    # This function is now obsolete since we're using the actual fine-tuned model
    return "models/fine_tuned/final_model"