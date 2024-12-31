import os
import typer
from typing import Optional, List
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
# Ensure PgVector is the correct class
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

# Assistant Function


def PDF_assistant(new: bool = False, user: str = "user"):
    run_id: Optional[str] = None

    # Retrieve existing run IDs
    if not new:
        existing_run_ids: List[str] = storage.get_all_run_ids(user)
        if len(existing_run_ids) > 0:
            run_id = existing_run_ids[0]

    # Initialize Assistant
    assistant = Assistant(
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        storage=storage,
        # Show tool calls in the response
        show_tool_calls=True,
        # Enable the assistant to search the knowledge base
        search_knowledge=True,
        # Enable the assistant to read the chat history
        read_chat_history=True,
    )

    # Start or continue the run
    if run_id is None:
        run_id = assistant.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    # Run the CLI application with Markdown support
    assistant.cli_app(markdown=True)


if __name__ == "__main__":
    typer.run(PDF_assistant)
