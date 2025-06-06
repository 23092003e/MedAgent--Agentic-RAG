import os
import sys
import time
from datetime import datetime
import requests
import logging
from pathlib import Path

# Add project root to path (MUST happen before backend imports!)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import streamlit as st
from backend.config import DB_FAISS_PATH
from backend.rag.vector_store import load_vector_store
from backend.rag.retrieval_qa import create_qa_chain, load_llm
from backend.rag.logging_config import setup_logging
from backend.rag.exceptions import MedAgentError, ConnectionError
from backend.rag.resource_manager import resource_manager
from backend.rag.self_reflection import SelfReflectionChain

# Setup logging
logger = setup_logging(Path("medagent.log"))

# Page configuration
st.set_page_config(
    page_title="MedAgent Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved appearance
def apply_custom_css():
    st.markdown("""
    <style>
        /* Main app styling */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Header styling */
        h1 {
            color: #1E88E5;
            font-family: 'Helvetica Neue', sans-serif;
            margin-bottom: 1rem;
        }
        
        /* Sidebar styling */
        .css-1d391kg, .css-1lcbmhc {
            background-color: #1A202C;
        }
        
        .sidebar .sidebar-content {
            background-color: #1A202C;
            color: white;
        }
        
        [data-testid="stSidebar"] {
            background-color: #1A202C;
            color: white;
        }
        
        [data-testid="stSidebar"] .sidebar-content {
            background-color: #1A202C;
        }
        
        [data-testid="stSidebarNav"] {
            background-color: #1A202C;
        }
        
        [data-testid="stSidebarNavItems"] {
            background-color: #1A202C;
        }
        
        /* Sidebar title */
        [data-testid="stSidebar"] h1 {
            color: #3B82F6;
            font-weight: bold;
        }
        
        [data-testid="stSidebar"] h2 {
            color: #E0E0E0;
        }
        
        [data-testid="stSidebar"] h3 {
            color: #E0E0E0;
        }
        
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            color: #E0E0E0;
        }
        
        /* Sidebar separator */
        [data-testid="stSidebar"] hr {
            border-color: #2D3748;
            margin: 20px 0;
        }
        
        /* Settings label */
        [data-testid="stSidebar"] .stSlider label, 
        [data-testid="stSidebar"] .stCheckbox label {
            color: #E0E0E0 !important;
        }
        
        /* Clear button */
        [data-testid="stSidebar"] .stButton button {
            background-color: #3B82F6;
            color: white;
            font-weight: bold;
            border-radius: 6px;
            border: none;
            padding: 0.5rem 1rem;
            width: 100%;
            transition: all 0.3s;
        }
        
        [data-testid="stSidebar"] .stButton button:hover {
            background-color: #2563EB;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Chat container styling */
        .chat-container {
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            background-color: #FFFFFF;
            height: auto;
            min-height: 200px;
            max-height: 65vh;  
            overflow-y: auto;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 20px;
        }
        
        /* User message styling */
        .user-message {
            background-color: #E3F2FD;
            border-radius: 18px 18px 0 18px;
            padding: 12px 18px;
            margin: 12px 0;
            max-width: 80%;
            margin-left: auto;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            color: #0D47A1;
        }
        
        /* Bot message styling */
        .bot-message {
            background-color: #F5F7FA;
            border-radius: 18px 18px 18px 0;
            padding: 12px 18px;
            margin: 12px 0;
            max-width: 80%;
            margin-right: auto;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            border-left: 4px solid #1E88E5;
            color: #333333;
        }
        
        /* Message icon */
        .message-icon {
            display: inline-block;
            width: 28px;
            height: 28px;
            line-height: 28px;
            text-align: center;
            border-radius: 50%;
            margin-right: 8px;
            font-size: 14px;
        }
        
        .user-icon {
            background-color: #E3F2FD;
            color: #1E88E5;
        }
        
        .bot-icon {
            background-color: #1E88E5;
            color: white;
        }
        
        /* Source documents styling */
        .source-docs {
            font-size: 0.85rem;
            color: #666;
            border-top: 1px solid #eee;
            margin-top: 8px;
            padding-top: 8px;
        }
        
        .source-title {
            color: #1E88E5;
            font-weight: 500;
            margin-bottom: 5px;
        }
        
        /* Input box styling */
        .input-container {
            display: flex;
            margin-top: 20px;
            background-color: white;
            border-radius: 20px;
            padding: 6px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            border: 1px solid #E0E0E0;
        }
        
        .stTextInput > div > div > input {
            border-radius: 20px;
            border: none;
            padding-left: 15px;
            background-color: transparent;
            color: #00000;
            box-shadow: none;
        }
        
        .stTextInput> div > div > input {
            color: #99999;
            opacity: 1;
        }
        
        .stTextInput > div > div > input:focus {
            color: #000000;
            background-color: #f9f9f9; 
        }
        
        /* Hide default Streamlit input border */
        .stTextInput > div {
            border: none !important;
            box-shadow: none !important;
        }
        
        /* Button styling */
        .send-button button {
            border-radius: 20px;
            background-color: #1E88E5;
            color: white;
            font-weight: 500;
            border: none;
            padding: 0.5rem 1rem;
            transition: all 0.3s;
        }
        
        .send-button button:hover {
            background-color: #1565C0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Timestamp styling */
        .timestamp {
            font-size: 0.7rem;
            color: #999;
            margin-bottom: 2px;
        }
        
        /* Fix for sidebar text color in dark theme */
        .sidebar .sidebar-content {
            color: white;
        }
        
        /* Footer styling */
        .footer {
            font-size: 0.8rem;
            color: #666;
            text-align: center;
            padding: 20px 0;
            border-top: 1px solid #eee;
            margin-top: 30px;
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_vectorstore():
    try:
        # Get from resource manager or create new
        vectorstore = resource_manager.get("vectorstore")
        if not vectorstore:
            vectorstore = load_vector_store(DB_FAISS_PATH)
            resource_manager.register("vectorstore", vectorstore)
        return vectorstore
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        return None

def handle_input():
    user_question = st.session_state.user_input
    if user_question:
        st.session_state.messages.append({
            'role': 'user', 
            'content': user_question,
            'timestamp': datetime.now().strftime("%H:%M")
        })
        # Don't clear the input field here - we'll do it in the next rerun
        st.session_state.process_input = True  # Flag to process input in next rerun

def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🏥"):
            st.markdown(message["content"])
            
            # Display reflection analysis if available
            if message["role"] == "assistant" and "reflection" in message:
                reflection = message["reflection"]
                if reflection:
                    with st.expander("Analysis", label_visibility="visible"):
                        # Display confidence score with progress bar
                        confidence = reflection.get("confidence_score", 0)
                        st.progress(confidence/100, text=f"Confidence: {confidence}%")
                        
                        # Display verified claims
                        if reflection.get("verified_claims"):
                            st.markdown("**✓ Verified claims:**")
                            for claim in reflection["verified_claims"]:
                                st.markdown(f"- {claim}")
                                
                        # Display missing information
                        if reflection.get("missing_information"):
                            st.markdown("**ℹ️ Missing information:**")
                            for info in reflection["missing_information"]:
                                st.markdown(f"- {info}")
                                
                        # Display suggested improvements
                        if reflection.get("suggested_improvements"):
                            st.markdown("**↗️ Suggested improvements:**")
                            for improvement in reflection["suggested_improvements"]:
                                st.markdown(f"- {improvement}")
            
            # Display timestamp with proper label
            st.markdown(
                f"<div class='timestamp' aria-label='Message timestamp'>{message['timestamp']}</div>",
                unsafe_allow_html=True
            )

def clear_conversation():
    st.session_state.messages = []
    # Add welcome message back
    welcome_msg = {
        'role': 'assistant',
        'content': "👋 Hello! I'm MedAgent, your medical information assistant. How can I help you today?\n\n**Source Docs:**\nInternal knowledge base",
        'timestamp': datetime.now().strftime("%H:%M")
    }
    st.session_state.messages.append(welcome_msg)

def check_ollama_server():
    """Check if Ollama server is running and accessible"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        return response.status_code == 200
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Failed to connect to Ollama server: {e}")
        raise ConnectionError("Cannot connect to Ollama server") from e

def initialize_llm():
    """Initialize LLM with connection check"""
    try:
        if not check_ollama_server():
            raise ConnectionError(
                "Cannot connect to Ollama server. Please ensure:\n"
                "1. Ollama is installed\n"
                "2. Server is running (run 'ollama serve')\n"
                "3. Model is downloaded (run 'ollama pull medllama2')"
            )
        llm = load_llm()
        resource_manager.register("llm", llm)
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise

def main():
    try:
        apply_custom_css()
        
        # Initialize components
        llm = initialize_llm()
        if not llm:
            st.error("Failed to initialize AI model. Please check the logs.")
            return
            
        reflection_chain = SelfReflectionChain(llm)
        resource_manager.register("reflection_chain", reflection_chain)
        
        # Sidebar
        with st.sidebar:
            st.title("MedAgent")
            st.markdown("Your AI-powered medical information assistant")
            st.markdown("---")
            
            # Optional settings
            st.subheader("Settings")
            
            # We'll store these in session_state so they persist
            if 'temperature' not in st.session_state:
                st.session_state.temperature = 0.7
            if 'include_sources' not in st.session_state:
                st.session_state.include_sources = True
            
            temperature = st.slider("AI Creativity", 
                                   min_value=0.0, 
                                   max_value=1.0, 
                                   value=st.session_state.temperature, 
                                   step=0.1,
                                   key="temperature_slider")
            include_sources = st.checkbox("Show sources", 
                                         value=st.session_state.include_sources,
                                         key="include_sources_checkbox")
            
            st.markdown("---")
            st.button("Clear Conversation", on_click=clear_conversation)
            
            st.markdown("---")
            st.markdown("© 2025 MedAgent AI")
            st.markdown("For informational purposes only. Not a substitute for professional medical advice.")
        
        # Main content area
        col1, col2, col3 = st.columns([1, 6, 1])
        
        with col2:
            st.title("🏥 MedAgent Chatbot")
            st.markdown("Ask any medical question and get AI-powered answers based on trusted sources")
            
            # Initialize chat history
            if 'messages' not in st.session_state:
                st.session_state.messages = []
                
                # Add welcome message
                welcome_msg = {
                    'role': 'assistant',
                    'content': "👋 Hello! I'm MedAgent, your medical information assistant. How can I help you today?\n\n**Source Docs:**\nInternal knowledge base",
                    'timestamp': datetime.now().strftime("%H:%M")
                }
                st.session_state.messages.append(welcome_msg)
                
            # Initialize process_input flag if not exists
            if 'process_input' not in st.session_state:
                st.session_state.process_input = False
            
            # Handle input processing
            if st.session_state.process_input:
                st.session_state.process_input = False
                
                last_user_msg = next(
                    (msg for msg in reversed(st.session_state.messages) 
                    if msg['role'] == 'user'), 
                    None
                )
                
                if last_user_msg:
                    user_question = last_user_msg['content']
                    
                    with st.spinner("Thinking..."):
                        try:
                            # Get vector store
                            vectorstore = get_vectorstore()
                            if not vectorstore:
                                raise MedAgentError("Failed to access knowledge base")
                                
                            # Create and use QA chain
                            qa_chain = create_qa_chain(vectorstore)
                            response = qa_chain({'query': user_question})
                            
                            # Get answer and sources
                            answer = response.get('result', '') or "I couldn't generate an answer."
                            sources = response.get('source_documents', [])
                            
                            # Process response and reflection
                            if sources:
                                source_texts = [
                                    doc.page_content 
                                    for doc in sources 
                                    if hasattr(doc, 'page_content') and doc.page_content
                                ]
                                
                                if source_texts:
                                    reflection = reflection_chain.analyze_response(answer, source_texts)
                                else:
                                    reflection = {'analysis': {}, 'improved_response': None}
                            else:
                                reflection = {'analysis': {}, 'improved_response': None}
                            
                            # Format response
                            content = reflection.get('improved_response') or answer
                            
                            if st.session_state.include_sources and sources:
                                valid_sources = []
                                for doc in sources:
                                    if hasattr(doc, 'metadata'):
                                        source = doc.metadata.get('source', '')
                                        page = doc.metadata.get('page', '')
                                        section = doc.metadata.get('section', '')
                                        
                                        reference = f"- {source}"
                                        if page:
                                            reference += f" (Page {page})"
                                        elif section:
                                            reference += f" (Section {section})"
                                            
                                        # Add score if available for relevance indication
                                        score = doc.metadata.get('score', '')
                                        if score:
                                            reference += f" [Relevance: {score:.2f}]"
                                            
                                        valid_sources.append(reference)
                                
                                if valid_sources:
                                    source_text = '\n'.join(valid_sources)
                                    content = f"{content}\n\n**References:**\n{source_text}"
                            
                            # Add to chat history
                            assistant_msg = {
                                'role': 'assistant',
                                'content': content,
                                'timestamp': datetime.now().strftime("%H:%M"),
                                'reflection': reflection.get('analysis', {}) if reflection else {},
                                'sources': valid_sources if st.session_state.include_sources else []
                            }
                            st.session_state.messages.append(assistant_msg)
                            st.session_state.user_input = ""
                            
                        except ConnectionError as e:
                            st.error("Cannot connect to the AI model. Please ensure Ollama server is running.")
                            logger.error(f"Connection error: {e}")
                        except Exception as e:
                            st.error("An error occurred while processing your question. Please try again.")
                            logger.error(f"Error in processing: {e}")
            
            # Display chat messages
            display_chat_history()
            
            # Custom input area
            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            cols = st.columns([8, 1])
            with cols[0]:
                # Create the text input with on_change handler
                st.text_input("", 
                             placeholder="Type your question here...",
                             key="user_input", 
                             on_change=handle_input,
                             label_visibility="collapsed")
            with cols[1]:
                st.markdown('<div class="send-button">', unsafe_allow_html=True)
                st.button("Send", on_click=handle_input)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Footer
            st.markdown('<div class="footer">This AI assistant provides information for educational purposes only. Always consult with healthcare professionals for medical advice 🪄.</div>', unsafe_allow_html=True)

    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error("An unexpected error occurred. Please check the logs.")
    finally:
        # Cleanup resources on exit
        resource_manager.cleanup()

if __name__ == '__main__':
    main()