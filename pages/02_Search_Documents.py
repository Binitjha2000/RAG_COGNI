import streamlit as st
import os
from dotenv import load_dotenv

from vector_db import VectorDatabaseManager
from llm_access_control import AccessControlledLLM
from pia_access_control import PIAAccessControl

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Search Documents",
    page_icon="🔍",
    layout="wide"
)

# Main title
st.title("Search Documents")
st.markdown("Search across your documents using natural language queries.")

# Authentication check
if not st.session_state.get("user_role"):
    st.warning("Please login on the home page to access this functionality.")
    st.stop()

# Check if documents have been processed
if not st.session_state.get("processed_docs", False) or not st.session_state.get("vectorstore"):
    st.warning("No documents have been processed yet. Please upload and process documents first.")
    st.stop()

# Initialize access control
pia_access = PIAAccessControl()

# Query input
st.subheader("Search Query")
query = st.text_input("Enter your search query")

# Advanced search settings
with st.expander("Advanced Search Settings", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)
    with col2:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    chain_type = st.selectbox(
        "Chain Type",
        ["stuff", "map_reduce", "refine"],
        index=0,
        help="The method used to combine document information with your query"
    )

# Search button
if query:
    if st.button("Search"):
        with st.spinner("Searching documents..."):
            try:
                # Initialize LLM with access control
                llm_controller = AccessControlledLLM(temperature=temperature)
                
                # Get vector store retriever
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": top_k})
                
                # Create QA chain
                qa_chain = llm_controller.create_qa_chain(retriever, chain_type=chain_type)
                
                # Execute query with access control
                result = llm_controller.query_with_access_control(
                    qa_chain=qa_chain,
                    query=query,
                    user_role=st.session_state.user_role
                )
                
                # Display answer
                st.subheader("Answer")
                st.markdown(result["result"])
                
                # Display model information
                if "model_info" in result:
                    model_info = result["model_info"]
                    with st.expander("Model Information", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Model", f"{model_info.get('name', 'Unknown')}")
                        with col2:
                            st.metric("Type", f"{model_info.get('type', 'Unknown')}")
                        if "device" in model_info:
                            with col3:
                                st.metric("Device", f"{model_info.get('device', 'Unknown')}")
                        
                        if "quantization" in model_info:
                            st.info(f"Using {model_info.get('quantization', 'Unknown')} quantization")
                
                # Display source documents with access control filtering
                st.subheader("Sources")
                
                for i, doc in enumerate(result["source_documents"]):
                    # Check if user has access to this document
                    if pia_access.can_access_document(
                        user_role=st.session_state.user_role,
                        document_metadata=doc.metadata,
                        user_department=st.session_state.get("user_department")
                    ):
                        with st.expander(f"Source {i+1}"):
                            st.markdown(f"**Content:**")
                            st.markdown(doc.page_content)
                            
                            # Display metadata
                            st.markdown("**Metadata:**")
                            source = doc.metadata.get("source", "Unknown")
                            classification = doc.metadata.get("classification", "Unknown")
                            department = doc.metadata.get("department", "Unknown")
                            
                            st.markdown(f"- **Source:** {os.path.basename(source)}")
                            st.markdown(f"- **Classification:** {classification}")
                            st.markdown(f"- **Department:** {department}")
                            
                            # Display page info if available
                            if "page" in doc.metadata:
                                st.markdown(f"- **Page:** {doc.metadata['page']}")
                    else:
                        with st.expander(f"Source {i+1} (Access Restricted)"):
                            st.warning("You do not have permission to view this source document.")
            except Exception as e:
                st.error(f"Error during search: {str(e)}")

# Search history section
st.subheader("Recent Searches")

# Initialize search history in session state if it doesn't exist
if "search_history" not in st.session_state:
    st.session_state.search_history = []

# Display search history (most recent first)
if st.session_state.search_history:
    for i, history_item in enumerate(reversed(st.session_state.search_history[-5:])):
        with st.expander(f"{history_item['query']}", expanded=False):
            st.markdown(f"**Query:** {history_item['query']}")
            st.markdown(f"**Date:** {history_item['timestamp']}")
            st.markdown(f"**Role:** {history_item['user_role']}")
            
            if st.button(f"Rerun Query", key=f"rerun_{i}"):
                # Set the query input to the historical query
                st.session_state.query = history_item['query']
                st.rerun()
else:
    st.write("No recent searches.")

# Add current query to search history when executed
if query and st.button("Search") and not st.session_state.get("processed_docs", False):
    import time
    
    history_item = {
        "query": query,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "user_role": st.session_state.user_role,
        "user": st.session_state.get("user_name", "Anonymous")
    }
    
    st.session_state.search_history.append(history_item)
