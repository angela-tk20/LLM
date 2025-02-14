import os
import logging
from typing import List, Tuple, Optional

import streamlit as st
from dotenv import load_dotenv
from cassandra.cluster import Cluster, Session
from cassandra.auth import PlainTextAuthProvider
from cassandra.policies import DCAwareRoundRobinPolicy
from google.generativeai import GenerativeModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Configuration
KEYSPACE_NAME = "erfrrgedrthdsrtgq354w6w5867werafgsdfert7urutfjd6"
COLLECTION_NAME = "grahams_db"
MAX_RESULTS = 5
SECURE_BUNDLE_PATH = "secure-connect-my-database.zip"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, secure_bundle_path: str):
        """Initialize database connection with the secure bundle."""
        self.session = self._initialize_connection(secure_bundle_path)
        logger.info("Database connection initialized")

    def _initialize_connection(self, secure_bundle_path: str) -> Session:
        """Create and return a database session."""
        try:
            cloud_config = {'secure_connect_bundle': secure_bundle_path}
            auth_provider = PlainTextAuthProvider(
                os.getenv('ASTRA_CLIENT_ID'),
                os.getenv('ASTRA_CLIENT_SECRET')
            )
            cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider,protocol_version=4, connect_timeout=60,  # Increase connection timeout
            control_connection_timeout=60,executor_threads=4,metrics_enabled=False,idle_heartbeat_interval=30)
            session = cluster.connect(wait_for_all_pools=False)
            session.set_keyspace(KEYSPACE_NAME)
            return session
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise

    def verify_table_exists(self) -> bool:
        """Verify if the table exists in the keyspace and show its schema."""
        try:
            # Check if table exists
            query = """
               SELECT table_name 
               FROM system_schema.tables 
               WHERE keyspace_name = %s 
               AND table_name = %s;
               """
            rows = self.session.execute(query, [KEYSPACE_NAME, COLLECTION_NAME])
            exists = len(list(rows)) > 0

            if exists:
                # Get table schema
                schema_query = f"DESCRIBE TABLE {KEYSPACE_NAME}.{COLLECTION_NAME};"
                schema = self.session.execute(schema_query)
                logger.info(f"Table schema: {list(schema)}")
                return True
            else:
                logger.warning(f"Table {COLLECTION_NAME} does not exist in keyspace {KEYSPACE_NAME}")
                return False
        except Exception as e:
            logger.error(f"Error checking table: {str(e)}")
            raise

    def test_table_access(self)-> bool:
        """Test if we can read from the table."""
        try:
            query = f"SELECT * FROM {COLLECTION_NAME} LIMIT 1;"
            rows = self.session.execute(query)
            results = list(rows)
            logger.info(f"Test query returned {len(results)} rows")
            if results:
                logger.info("Sample row columns: " + ", ".join(results[0]._fields))
            return bool(results)
        except Exception as e:
            logger.error(f"Error testing table access: {str(e)}")
            raise

    def search_similar_chunks(self, query_embedding: List[float]) -> List[Tuple[str, float]]:
        """Search for similar document chunks using vector similarity."""
        try:
            query = """
            SELECT document_text, (embedding, %s) AS similarity
            FROM {} 
            ORDER BY similarity DESC
            LIMIT %s
            """.format(COLLECTION_NAME)

            parameters = [list(query_embedding), MAX_RESULTS]
            logger.info(f"Executing query with {MAX_RESULTS} limit")

            prepared_embedding = query_embedding
            rows = self.session.execute(query, [list(query_embedding)])

            results = [(row.document_text, float(row.similarity)) for row in rows]
            logger.info(f"Found {len(results)} matching documents")

            return results
        except Exception as e:
            logger.error(f"Error during vector search: {str(e)}")
            logger.error(f"Query parameters: limit={MAX_RESULTS}, embedding_size={len(query_embedding)}")
            raise


class QueryEngine:
    def __init__(self, db_manager: DatabaseManager):
        """Initialize the query engine with database manager and AI models."""
        self.db_manager = db_manager
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv('GOOGLE_API_KEY')
        )
        self.model = GenerativeModel('gemini-1.0-pro')
        logger.info("Query engine initialized")

    def process_query(self, question: str) -> Tuple[str, List[Tuple[str, float]]]:
        """Process a user query and return the answer with relevant sources."""
        try:
            query_embedding = self.embeddings.embed_query(question)
            relevant_docs = self.db_manager.search_similar_chunks(query_embedding)

            if not relevant_docs:
                return "I couldn't find any relevant information in my knowledge base.", []

            context = "\n".join([doc[0] for doc in relevant_docs])
            prompt = self._construct_prompt(context, question)
            response = self.model.generate_content(prompt)

            return response.text, relevant_docs
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    def _construct_prompt(self, context: str, question: str) -> str:
        """Construct the prompt for the AI model."""
        return f"""
        I am an AI that responds to questions related to Paul Graham.
        Based on the following context, provide a detailed answer to the question.
        If the information cannot be found in the context, clearly state that.
        Keep answers focused on Paul Graham's essays, work, and ideas.

        Context: {context}

        Question: {question}
        """


def setup_streamlit():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Graham's Chatbot",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    st.title("Graham's Chatbot")


def main():
    """Main application entry point."""
    load_dotenv()
    setup_streamlit()

    try:
        db_manager = DatabaseManager(SECURE_BUNDLE_PATH)

        if not db_manager.verify_table_exists():
            st.error(f"Table {COLLECTION_NAME} not found in database!")
            return

        if not db_manager.test_table_access():
            st.warning("Table exists but appears to be empty")
            return

        query_engine = QueryEngine(db_manager)
        st.success("Connected to Database")

        # Add description
        st.markdown("""
        This chatbot can answer questions about Paul Graham's essays and ideas.
        Ask any question about his writings, startup philosophy, or tech insights.
        """)

        # Query input
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g., What does Paul Graham think about startups?"
        )

        if question:
            with st.spinner("Searching for relevant information..."):
                answer, relevant_docs = query_engine.process_query(question)

                st.subheader("Answer")
                st.write(answer)

                if relevant_docs:
                    with st.expander("View Source Documents"):
                        for doc, similarity in relevant_docs:
                            st.markdown(f"**Relevance Score: {(1 - similarity):.2%}**")
                            st.markdown(doc)
                            st.divider()

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An error occurred. Please try again later or contact support.")


if __name__ == "__main__":
    main()