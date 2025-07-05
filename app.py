# app.py

# Import from new, specific LangChain packages
from langchain_huggingface.llms import HuggingFacePipeline # Updated import
from langchain_huggingface.embeddings import HuggingFaceEmbeddings # Will need this for RAG if we move embedding to app.py, but for now just for illustration of new import style
from langchain_chroma import Chroma # Will need this for RAG if we move Chroma to app.py

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory # Still from langchain.memory for now

# Import our modular components
from llm_manager import LLMManager
from rag_system import RAGSystem
import config

def main():
    print("--- Initializing Local LLM Chatbot Application ---")

    # 1. Initialize LLM Manager
    llm_manager = LLMManager()
    transformers_pipeline = llm_manager.get_llm_pipeline()
    tokenizer = llm_manager.get_tokenizer()
    device = llm_manager.get_device()

    # Create the LangChain-compatible HuggingFacePipeline instance
    llm_for_langchain = HuggingFacePipeline(pipeline=transformers_pipeline)

    # 2. Initialize RAG System
    # Pass the device to RAGSystem for embedding model
    rag_system = RAGSystem(device=device)
    retriever = rag_system.get_retriever()

    # 3. Set up LangChain Conversational Retrieval Chain
    # Set 'output_key' to 'answer' to tell memory which part to store
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer') # ADDED output_key

    qa_chain = None
    if retriever:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm_for_langchain,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            # Ensure the output key is explicitly set if the chain has multiple outputs
            # This is often handled by the memory or chain itself, but good to be aware
            # For ConversationalRetrievalChain, 'answer' is typically the default output.
            # The error suggests memory is confused, so setting it in memory is better.
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
                # The invoke method returns a dictionary, 'answer' is the key for the response
                result = qa_chain.invoke({"question": user_input})
                response_text = result["answer"]
                print(f"Bot: {response_text}")

                if result.get("source_documents"):
                    print("\nSources:")
                    for i, doc in enumerate(result["source_documents"]):
                        print(f"  {i+1}. Content: {doc.page_content[:150]}... (Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')})")
                    print("-" * 20)
            else:
                if tokenizer is None or transformers_pipeline is None:
                    print("Error: LLM not loaded, cannot proceed with general chat.")
                    continue

                chat_history_for_llm = memory.chat_memory.messages + [{"role": "user", "content": user_input}]
                formatted_input = tokenizer.apply_chat_template(
                    chat_history_for_llm,
                    tokenize=False,
                    add_generation_prompt=True
                )
                response_text = transformers_pipeline(formatted_input)[0]['generated_text']
                print(f"Bot: {response_text}")
                # Manually add to memory for LLM-only mode
                memory.chat_memory.add_user_message(user_input)
                memory.chat_memory.add_ai_message(response_text)

        except Exception as e:
            print(f"Error during chatbot interaction: {e}")
            print("Please ensure your documents are correctly placed and the model is loaded.")

    print("\n--- Conversation Ended ---")

    llm_manager.save_model_and_tokenizer()

if __name__ == "__main__":
    main()