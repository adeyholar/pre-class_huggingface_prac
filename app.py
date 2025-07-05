# app.py

# Ensure HuggingFacePipeline is directly imported
from langchain_community.llms import HuggingFacePipeline

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Import our modular components
from llm_manager import LLMManager
from rag_system import RAGSystem # Ensure this line is exactly as shown
import config

def main():
    print("--- Initializing Local LLM Chatbot Application ---")

    llm_manager = LLMManager()
    llm_pipeline_instance = llm_manager.get_llm_pipeline()
    tokenizer = llm_manager.get_tokenizer()
    device = llm_manager.get_device()

    rag_system = RAGSystem(device=device)
    retriever = rag_system.get_retriever()

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = None
    if retriever:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm_pipeline_instance,
            retriever=retriever,
            memory=memory,
            return_source_documents=True
        )
        print("RAG chain initialized. You can now ask questions about your documents.")
    else:
        print("No RAG documents loaded. Falling back to general chat mode (LLM only).")

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
                if tokenizer is None or llm_pipeline_instance is None:
                    print("Error: LLM not loaded, cannot proceed with general chat.")
                    continue

                chat_history_for_llm = memory.chat_memory.messages + [{"role": "user", "content": user_input}]
                formatted_input = tokenizer.apply_chat_template(
                    chat_history_for_llm,
                    tokenize=False,
                    add_generation_prompt=True
                )
                response_text = llm_pipeline_instance(formatted_input)[0]['generated_text']
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