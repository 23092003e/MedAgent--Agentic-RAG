import os
import sys
import time
from datetime import datetime
import requests
import logging

# Add project root to path (MUST happen before backend imports!)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import streamlit as st
from backend.config import DB_FAISS_PATH
from backend.rag.vector_store import load_vector_store
from backend.rag.retrieval_qa import create_qa_chain

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
        return load_vector_store(DB_FAISS_PATH)
    except Exception as e:
        st.error(f"Failed to load vector store: {e}")
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
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            timestamp = msg.get('timestamp', datetime.now().strftime("%H:%M"))
            
            if msg['role'] == 'user':
                st.markdown(f'<div class="timestamp" style="text-align: right;">{timestamp}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="user-message"><span class="message-icon user-icon">👤</span>{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                # Split content and reflection
                content_parts = msg['content'].split("**Source Docs:**")
                answer = content_parts[0].strip()
                
                st.markdown(f'<div class="bot-message">', unsafe_allow_html=True)
                st.markdown(f'<span class="message-icon bot-icon">🏥</span>{answer}', unsafe_allow_html=True)
                
                # Display confidence score if available
                if 'reflection' in msg:
                    confidence = msg['reflection'].get('confidence_score', 0)
                    st.progress(confidence/100, text=f"Confidence: {confidence}%")
                    
                    # Show reflection details in expander
                    with st.expander("See analysis"):
                        st.write("Verified claims:", msg['reflection'].get('verified_claims', []))
                        st.write("Missing information:", msg['reflection'].get('missing_information', []))
                        st.write("Suggested improvements:", msg['reflection'].get('suggested_improvements', []))
                
                # Display sources
                if len(content_parts) > 1 and st.session_state.include_sources:
                    sources = content_parts[1].strip()
                    st.markdown(f'<div class="source-docs"><div class="source-title">Sources:</div>{sources}</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def clear_conversation():
    st.session_state.messages = []
    # Add welcome message back
    welcome_msg = {
        'role': 'assistant',
        'content': "👋 Hello! I'm MedAgent, your medical information assistant. How can I help you today?\n\n**Source Docs:**\nInternal knowledge base",
        'timestamp': datetime.now().strftime("%H:%M")
    }
    st.session_state.messages.append(welcome_msg)

def main():
    apply_custom_css()
    
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
            
        # Handle any pending input processing from previous run
        if st.session_state.process_input:
            # Reset the flag first to avoid infinite loop
            st.session_state.process_input = False
            
            # Get the last user message
            last_user_msg = next((msg for msg in reversed(st.session_state.messages) 
                                if msg['role'] == 'user'), None)
            
            if last_user_msg:
                user_question = last_user_msg['content']
                
                # Show loading spinner
                with st.spinner("Thinking..."):
                    vectorstore = get_vectorstore()
                    if not vectorstore:
                        st.error("Failed to access the knowledge base. Please try again later.")
                    else:
                        try:
                            # Create and invoke QA chain
                            qa_chain = create_qa_chain(vectorstore)
                            response = qa_chain.invoke({'query': user_question})
                            answer = response.get('result', '')
                            
                            # Process sources based on settings
                            sources = response.get('source_documents', [])
                            source_text = ''
                            if st.session_state.include_sources and sources:
                                source_text = '\n'.join([f"- {doc.metadata.get('source', '')}" for doc in sources])
                            
                            # Format the assistant's response
                            content = f"{answer}"
                            if source_text:
                                content += f"\n\n**Source Docs:**\n{source_text}"
                            
                            # Simulate typing delay for a more natural feel
                            time.sleep(0.5)
                            
                            # Add assistant message to chat history
                            assistant_msg = {
                                'role': 'assistant', 
                                'content': content,
                                'timestamp': datetime.now().strftime("%H:%M")
                            }
                            st.session_state.messages.append(assistant_msg)
                            
                            # Clear the input field for next input
                            st.session_state.user_input = ""
                        except requests.exceptions.ConnectionError:
                            st.error("Cannot connect to the AI model. Please ensure Ollama server is running.")
                            logger.error("Failed to connect to Ollama server at localhost:11434")
                        except Exception as e:
                            st.error("An error occurred while processing your question. Please try again.")
                            logger.error(f"Error in QA chain: {str(e)}")
        
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

if __name__ == '__main__':
    main()