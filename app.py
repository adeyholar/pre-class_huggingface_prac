# app.py

# Ensure HuggingFacePipeline is directly imported
from langchain_community.llms import HuggingFacePipeline # ADDED this line explicitly

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Import our modular components
from llm_manager import LLMManager
from rag_system import RAGSystem
import config # For configuration settings

def main():
    print("--- Initializing Local LLM Chatbot Application ---")

    # 1. Initialize LLM Manager
    llm_manager = LLMManager()
    llm_pipeline_instance = llm_manager.get_llm_pipeline() # Renamed to avoid conflict with class name
    tokenizer = llm_manager.get_tokenizer()
    device = llm_manager.get_device()

    # 2. Initialize RAG System
    rag_system = RAGSystem(device=device)
    retriever = rag_system.get_retriever()

    # 3. Set up LangChain Conversational Retrieval Chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = None # Initialize qa_chain to None
    if retriever: # Only set up RAG chain if retriever is available (i.e., documents were loaded)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm_pipeline_instance, # Use the renamed instance
            retriever=retriever,
            memory=memory,
            return_source_documents=True
        )
        print("RAG chain initialized. You can now ask questions about your documents.")
    else:
        print("No RAG documents loaded. Falling back to general chat mode (LLM only).")

    # --- Interactive Chat Loop ---
    print("\n--- Start Chatting! Type 'exit' to quit. ---")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        try:
            if qa_chain:
                result = qa_chain.invoke({"question": user_input})
                response_text = result["answer"]
                print(f"Bot: {response_text}")

                if result.get("source_documents"):
                    print("\nSources:")
                    for i, doc in enumerate(result["source_documents"]):
                        print(f"  {i+1}. Content: {doc.page_content[:150]}... (Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')})")
                    print("-" * 20)
            else:
                # If RAG is not set up, use the LLM pipeline directly for general chat
                # Ensure tokenizer and llm_pipeline_instance are not None
                if tokenizer is None or llm_pipeline_instance is None:
                    print("Error: LLM not loaded, cannot proceed with general chat.")
                    continue

                chat_history_for_llm = memory.chat_memory.messages + [{"role": "user", "content": user_input}]
                formatted_input = tokenizer.apply_chat_template( # Pylance should now resolve this
                    chat_history_for_llm,
                    tokenize=False,
                    add_generation_prompt=True
                )
                response_text = llm_pipeline_instance(formatted_input)[0]['generated_text'] # Pylance should now resolve this
                print(f"Bot: {response_text}")
                memory.chat_memory.add_user_message(user_input)
                memory.chat_memory.add_ai_message(response_text)

        except Exception as e:
            print(f"Error during chatbot interaction: {e}")
            print("Please ensure your documents are correctly placed and the model is loaded.")

    print("\n--- Conversation Ended ---")

    llm_manager.save_model_and_tokenizer()

if __name__ == "__main__":
    main()