import os
from typing import Optional
import streamlit as st
from phi.assistant import Assistant
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Database URL
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# Initialize vector database
vector_db = PgVector(table_name="recipes", db_url=db_url)

# Knowledge Base Initialization
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=vector_db
)

# Load the knowledge base: Uncomment and run this once to load the PDF
# knowledge_base.load()

# Storage setup
storage = PgAssistantStorage(table_name="PDF_assistant", db_url=db_url)

# Initialize Assistant
assistant = Assistant(
    run_id=None,  # Start a new session by default
    user_id="test_user",  # Replace with the actual user ID if applicable
    knowledge_base=knowledge_base,
    storage=storage,
    show_tool_calls=True,
    search_knowledge=True,
    read_chat_history=True,
)

# Streamlit App
st.title("Chat with the Assistant")

# Display session info
if assistant.run_id is None:
    st.write("Started a new session!")
else:
    st.write(f"Continuing session: `{assistant.run_id}`")

# Chat Interface
chat_input = st.text_input(
    "Your question:", placeholder="Ask something about Thai Recipes...")

if st.button("Ask"):
    if chat_input.strip():
        try:
            # Use the correct method to interact with the assistant
            # Use `chat` or the correct method
            response = assistant.chat(chat_input)
            st.markdown("**Assistant Response:**")
            st.write(response)
        except AttributeError as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a valid question.")

# Option to view chat history
if st.checkbox("View Chat History"):
    try:
        chat_history = assistant.get_chat_history()
        st.subheader("Chat History")
        for message in chat_history:
            st.markdown(f"**{message['role']}**: {message['content']}")
    except Exception as e:
        st.error(f"Could not fetch chat history: {e}")

if __name__ == "__main__":
    # This ensures the script runs correctly
    st.write("Running Streamlit app!")
