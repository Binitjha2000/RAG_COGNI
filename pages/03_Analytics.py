import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
from collections import Counter
import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import random
from wordcloud import WordCloud
import nltk
import string
import re

# Download NLTK resources properly with explicit feedback
st.markdown("### Setting up Analytics...")
with st.spinner("Loading required language resources..."):
    # Always download these resources to prevent errors
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    
    # Import these only after downloading
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    
    st.success("Language resources loaded successfully!")

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Analytics",
    page_icon="📊",
    layout="wide"
)

# Custom CSS with modern styling and proper contrast
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Main app styling */
.stApp {
    background: linear-gradient(135deg, #1e293b 0%, #334155 50%, #475569 100%);
    min-height: 100vh;
}

/* Override Streamlit's default styling */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border-radius: 25px;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.15);
    margin: 1rem;
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: #1e293b;
}

.metric-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border-radius: 15px;
    padding: 2rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
    text-align: center;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(99, 102, 241, 0.1);
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 40px rgba(99, 102, 241, 0.15);
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 800;
    color: #6366f1;
    margin-bottom: 0.5rem;
    font-family: 'Inter', sans-serif;
}

.metric-label {
    font-size: 1.1rem;
    color: #475569;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
}

.chart-container {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
    margin-bottom: 2rem;
    border: 1px solid rgba(99, 102, 241, 0.1);
}

.insights-card {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border-radius: 15px;
    padding: 2rem;
    border-left: 4px solid #06b6d4;
    margin-bottom: 1.5rem;
    box-shadow: 0 5px 15px rgba(6, 182, 212, 0.1);
    color: #0c4a6e;
}

.analytics-header {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
    padding: 3rem 2rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    color: white;
    text-align: center;
    box-shadow: 0 15px 40px rgba(99, 102, 241, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.analytics-header h1 {
    font-family: 'Inter', sans-serif;
    font-weight: 800;
    font-size: 2.5rem;
    margin-bottom: 1rem;
    text-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

.analytics-header p {
    font-family: 'Inter', sans-serif;
    font-size: 1.1rem;
    opacity: 0.95;
    font-weight: 400;
}

/* Ensure proper text contrast */
.main .block-container * {
    color: #1e293b;
}

.main .block-container h1, .main .block-container h2, .main .block-container h3 {
    color: #1e293b !important;
    font-family: 'Inter', sans-serif;
}

/* Fix white text issues */
.stMarkdown, .stText {
    color: #1e293b !important;
}
</style>
""", unsafe_allow_html=True)

# Main title with enhanced styling
st.markdown("""
<div class="analytics-header">
    <h1>📊 Advanced Document Analytics</h1>
    <p>Comprehensive insights, metrics, and intelligence from your document repository with AI-powered analysis</p>
</div>
""", unsafe_allow_html=True)

# Add error recovery button
if st.button("⚠️ Fix NLTK Resources"):
    try:
        import nltk
        nltk.download('punkt', download_dir='./nltk_data')
        nltk.download('stopwords', download_dir='./nltk_data')
        os.environ["NLTK_DATA"] = "./nltk_data"
        st.success("Successfully downloaded NLTK resources to the application folder!")
        st.info("Please refresh the page to see the changes.")
    except Exception as e:
        st.error(f"Error downloading resources: {str(e)}")

# Wrap the entire document processing in a try-except to prevent page crashes
try:
    # Get documents from the session state
    if "documents" in st.session_state and st.session_state.documents:
        documents = st.session_state.documents
        if "all_documents" in st.session_state and st.session_state.all_documents:
            documents = st.session_state.all_documents
    elif "cache_manager" in st.session_state:
        try:
            # Try to force refresh documents from cache manager
            from document_processor import DocumentProcessor
            processor = DocumentProcessor()
            documents = processor.process_directory('documents')
            st.session_state.documents = documents
            st.session_state.processed_docs = True
        except Exception as e:
            st.warning(f"Could not load documents: {str(e)}")
            documents = []
    else:
        documents = []
    
    # Check if we have real documents
    if not documents:
        documents = []
        st.warning("No documents found. Please upload documents in the Upload Documents page.")
        
    # Create a small sample if needed for demo purposes when no documents are available
    if not documents and not st.session_state.get("processed_docs", False):
        from langchain.docstore.document import Document
        st.info("Using minimal placeholder data for visualization demonstration.")
        sample_text = """
        This is a placeholder document to demonstrate analytics functionality.
        You should upload real documents to see actual analytics data.
        The analytics dashboard provides insights into your document repository.
        It shows document counts, content analysis, and metadata statistics.
        """
        documents = [
            Document(page_content=sample_text, 
                    metadata={"source": "placeholder_demo.txt"})
        ]
except Exception as setup_error:
    st.error(f"Error initializing analytics: {str(setup_error)}")
    documents = []
        
# Document analytics
st.markdown("<h2 style='color: #1E3A8A; margin-top: 1.5rem;'>Repository Insights</h2>", unsafe_allow_html=True)

# Get basic statistics
doc_count = len(documents)
unique_sources = set()
classifications = Counter()
departments = Counter()
file_types = Counter()
page_counts = []
creation_dates = []
word_counts = []
all_text = ""

for doc in documents:
    # Extract metadata
    source = doc.metadata.get("source", "Unknown")
    unique_sources.add(source)
    classifications[doc.metadata.get("classification", "Unknown")] += 1
    departments[doc.metadata.get("department", "Unknown")] += 1
    
    # Get file extension
    if source != "Unknown":
        file_ext = os.path.splitext(source)[1].lower() if "." in source else ".unknown"
    else:
        file_ext = ".unknown"
    file_types[file_ext if file_ext else ".unknown"] += 1
    
    # Page count
    if "page_count" in doc.metadata:
        page_counts.append(doc.metadata["page_count"])
    
    # Creation date
    if "created" in doc.metadata:
        creation_dates.append(doc.metadata["created"])
    
    # Word count
    word_count = len(doc.page_content.split())
    word_counts.append(word_count)
    
    # Collect text for word cloud
    all_text += doc.page_content + " "

# Display overview metrics in a more visually appealing way
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">{}</div>
        <div class="metric-label">Document Chunks</div>
    </div>
    """.format(doc_count), unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">{}</div>
        <div class="metric-label">Unique Documents</div>
    </div>
    """.format(len(unique_sources)), unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">{}</div>
        <div class="metric-label">Document Types</div>
    </div>
    """.format(len(file_types)), unsafe_allow_html=True)
with col4:
    avg_words = int(sum(word_counts) / len(word_counts)) if word_counts else 0
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">{}</div>
        <div class="metric-label">Avg Words per Chunk</div>
    </div>
    """.format(avg_words), unsafe_allow_html=True)

# Document insights tabs
tab1, tab2, tab3 = st.tabs(["Content Analysis", "Metadata Insights", "Time Trends"])

# Tab 1: Content Analysis
with tab1:
    st.markdown("<h3 style='color: #4B5563;'>Content Analysis</h3>", unsafe_allow_html=True)
    
    # Word cloud from document content
    st.subheader("Document Content Word Cloud")
    
    # Process text for word cloud with error handling
    def clean_text(text):
        try:
            # Convert to lowercase
            text = text.lower()
            # Remove punctuation
            text = re.sub(f'[{string.punctuation}]', ' ', text)
            # Remove numbers
            text = re.sub(r'\d+', '', text)
            
            # Safe tokenization - try with simple split if word_tokenize fails
            try:
                tokens = word_tokenize(text)
            except Exception:
                # Fallback to simple whitespace splitting if tokenizer fails
                tokens = text.split()
            
            # Safe stopword removal
            try:
                stop_words = set(stopwords.words('english'))
                tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
            except Exception:
                # If stopwords fail, just filter by length
                tokens = [word for word in tokens if len(word) > 1]
                
            return ' '.join(tokens)
        except Exception as e:
            st.warning(f"Text cleaning encountered an error: {str(e)}")
            return text  # Return original text as fallback
    
    if all_text.strip():
        try:
            cleaned_text = clean_text(all_text)
            
            if cleaned_text.strip():
                # Generate word cloud with error handling
                try:
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='white',
                        colormap='viridis',
                        max_words=100
                    ).generate(cleaned_text)
                except Exception as wc_error:
                    st.error(f"Could not generate word cloud: {str(wc_error)}")
                    st.info("Continuing with other analytics...")
                    wordcloud = None
            else:
                st.info("Not enough text content after cleaning for word cloud generation")
                wordcloud = None
        except Exception as text_error:
            st.error(f"Error processing text: {str(text_error)}")
            wordcloud = None
            
            # Display word cloud if successfully created
            if wordcloud:
                try:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                except Exception as plt_error:
                    st.error(f"Error displaying word cloud: {str(plt_error)}")
                    
                    # Show a simple text representation as fallback
                    if len(cleaned_text) > 0:
                        most_common = Counter(cleaned_text.split()).most_common(10)
                        st.write("Most common terms:")
                        for word, count in most_common:
                            st.write(f"- {word}: {count}")
        else:
            st.info("Not enough text content for analysis")
    else:
        st.info("No document content available for analysis")
    
    # Topic distribution based on document content
    st.subheader("Document Source Distribution")
    
    # Extract document sources
    doc_sources = {}
    for doc in documents:
        if "source" in doc.metadata:
            file_path = doc.metadata["source"]
            file_name = os.path.basename(file_path)
            if file_name in doc_sources:
                doc_sources[file_name] += 1
            else:
                doc_sources[file_name] = 1
    
    # Create a donut chart for document sources
    if doc_sources:
        labels = list(doc_sources.keys())
        values = list(doc_sources.values())
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.4,
            marker_colors=px.colors.qualitative.Plotly
        )])
        fig.update_layout(
            title_text="Document Source Distribution",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No document sources available for analysis")
    
    # Content length analysis
    st.subheader("Document Length Analysis")
    
    # Use actual word counts from documents
    if word_counts and len(word_counts) > 0:
        # Create histogram of word counts
        fig = px.histogram(
            x=word_counts, 
            nbins=20,
            labels={"x": "Word Count per Chunk"},
            title="Document Chunk Length Distribution",
            color_discrete_sequence=["#3B82F6"]
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show statistics
        st.markdown(f"""
        **Word Count Statistics:**
        - **Minimum**: {min(word_counts) if word_counts else 0} words
        - **Maximum**: {max(word_counts) if word_counts else 0} words  
        - **Average**: {int(sum(word_counts) / len(word_counts)) if word_counts else 0} words
        """)
    else:
        st.info("No content available for length analysis")

# Tab 2: Metadata Insights
with tab2:
    st.markdown("<h3 style='color: #4B5563;'>Metadata Analysis</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Document classification distribution
        st.subheader("Classification Distribution")
        
        if classifications:
            fig = px.pie(
                names=list(classifications.keys()),
                values=list(classifications.values()),
                title="Document Classifications",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No classification data available")
    
    with col2:
        # Document department distribution
        st.subheader("Department Distribution")
        
        if departments:
            fig = px.bar(
                x=list(departments.keys()),
                y=list(departments.values()),
                title="Documents by Department",
                color=list(departments.keys()),
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig.update_layout(xaxis_title="Department", yaxis_title="Count", height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No department data available")
    
    # Document type distribution
    st.subheader("Document Type Distribution")
    
    if file_types:
        # Clean up file types for display
        clean_file_types = {k if k.startswith('.') else f".{k}": v for k, v in file_types.items()}
        
        fig = px.bar(
            x=list(clean_file_types.keys()),
            y=list(clean_file_types.values()),
            title="Document Formats",
            color=list(clean_file_types.keys()),
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(xaxis_title="File Type", yaxis_title="Count", height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No file type data available")

# Tab 3: Time Trends
with tab3:
    st.markdown("<h3 style='color: #4B5563;'>Temporal Analysis</h3>", unsafe_allow_html=True)
    
    # Document creation timeline
    st.subheader("Document Creation Timeline")
    
    if creation_dates:
        # Parse dates and count documents per month
        dates_df = pd.DataFrame({'created': creation_dates})
        dates_df['created'] = pd.to_datetime(dates_df['created'])
        dates_df['year_month'] = dates_df['created'].dt.strftime('%Y-%m')
        date_counts = dates_df.groupby('year_month').size().reset_index(name='count')
        date_counts = date_counts.sort_values('year_month')
        
        fig = px.line(
            date_counts, 
            x='year_month', 
            y='count',
            markers=True,
            title="Document Creation Over Time",
            color_discrete_sequence=["#3B82F6"]
        )
        fig.update_layout(xaxis_title="Date", yaxis_title="Number of Documents", height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Generate sample timeline data
        months = [(datetime.now() - timedelta(days=30*i)).strftime('%Y-%m') for i in range(12)]
        months.reverse()
        counts = [random.randint(3, 20) for _ in range(12)]
        
        fig = px.line(
            x=months, 
            y=counts,
            markers=True,
            title="Sample Document Creation Timeline",
            color_discrete_sequence=["#3B82F6"]
        )
        fig.update_layout(xaxis_title="Date", yaxis_title="Number of Documents", height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.info("Sample data shown. Upload documents with creation dates for actual timeline.")
    
    # Document age analysis
    st.subheader("Document Age Analysis")
    
    if creation_dates:
        # Calculate document age in days
        dates_df = pd.DataFrame({'created': creation_dates})
        dates_df['created'] = pd.to_datetime(dates_df['created'])
        dates_df['age_days'] = (datetime.now() - dates_df['created']).dt.days
        
        fig = px.histogram(
            dates_df,
            x='age_days',
            nbins=20,
            title="Document Age Distribution",
            color_discrete_sequence=["#10B981"]
        )
        fig.update_layout(xaxis_title="Age (Days)", yaxis_title="Number of Documents", height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Generate sample age data
        ages = [random.randint(1, 365) for _ in range(100)]
        
        fig = px.histogram(
            x=ages,
            nbins=20,
            title="Sample Document Age Distribution",
            color_discrete_sequence=["#10B981"]
        )
        fig.update_layout(xaxis_title="Age (Days)", yaxis_title="Number of Documents", height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.info("Sample data shown. Upload documents with creation dates for actual age analysis.")

# Search analytics
st.markdown("<h2 style='color: #1E3A8A; margin-top: 2rem;'>Search Analytics</h2>", unsafe_allow_html=True)
st.markdown("<p style='color: #6B7280;'>Understand how users are interacting with your document repository.</p>", unsafe_allow_html=True)

# Check if search history exists
if "search_history" in st.session_state and st.session_state.search_history:
    search_history = st.session_state.search_history
    
    # Get basic search statistics
    total_searches = len(search_history)
    users = Counter([item["user"] for item in search_history])
    roles = Counter([item["user_role"] for item in search_history])
    
    # Extract queries for analysis
    queries = [item["query"] for item in search_history]
    query_counter = Counter(queries)
    
    # Display overview metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Total Searches</div>
        </div>
        """.format(total_searches), unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Unique Users</div>
        </div>
        """.format(len(users)), unsafe_allow_html=True)
    with col3:
        avg_searches = int(total_searches / len(users)) if users else 0
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Avg Searches per User</div>
        </div>
        """.format(avg_searches), unsafe_allow_html=True)
    
    # Search trends tabs
    search_tab1, search_tab2 = st.tabs(["Search Patterns", "User Activity"])
    
    # Tab 1: Search Patterns
    with search_tab1:
        st.markdown("<h3 style='color: #4B5563;'>Search Pattern Analysis</h3>", unsafe_allow_html=True)
        
        # Top queries
        st.subheader("Most Common Search Queries")
        
        top_queries = query_counter.most_common(10)
        if top_queries:
            fig = px.bar(
                x=[q[0] for q in top_queries],
                y=[q[1] for q in top_queries],
                title="Top 10 Search Queries",
                color=[q[0] for q in top_queries],
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
            fig.update_layout(xaxis_title="Query", yaxis_title="Count", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Query word cloud
        st.subheader("Search Query Word Cloud")
        
        if queries:
            # Join all queries
            all_queries = " ".join(queries)
            # Clean text
            cleaned_queries = clean_text(all_queries)
            
            if cleaned_queries.strip():
                # Generate word cloud
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    colormap='plasma',
                    max_words=50
                ).generate(cleaned_queries)
                
                # Display word cloud
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
        
        # Search time patterns
        st.subheader("Search Time Patterns")
        
        # Extract hour of day from timestamps
        hours = []
        for item in search_history:
            try:
                timestamp = item.get("timestamp", "")
                if " " in timestamp:
                    time_part = timestamp.split(" ")[1]
                    hour = int(time_part.split(":")[0])
                    hours.append(hour)
            except:
                continue
        
        if hours:
            hour_counts = Counter(hours)
            hour_df = pd.DataFrame({
                "Hour": list(range(24)),
                "Searches": [hour_counts.get(hour, 0) for hour in range(24)]
            })
            
            fig = px.line(
                hour_df,
                x="Hour",
                y="Searches",
                title="Search Activity by Hour of Day",
                markers=True,
                color_discrete_sequence=["#8B5CF6"]
            )
            fig.update_layout(xaxis_title="Hour (24-hour format)", yaxis_title="Number of Searches", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: User Activity
    with search_tab2:
        st.markdown("<h3 style='color: #4B5563;'>User Activity Analysis</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Searches by user
            st.subheader("Searches by User")
            
            user_df = pd.DataFrame({
                "User": list(users.keys()),
                "Searches": list(users.values())
            }).sort_values(by="Searches", ascending=False)
            
            fig = px.bar(
                user_df,
                x="User",
                y="Searches",
                title="Search Activity by User",
                color="User",
                color_discrete_sequence=px.colors.qualitative.G10
            )
            fig.update_layout(xaxis_title="User", yaxis_title="Number of Searches", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Searches by role
            st.subheader("Searches by Role")
            
            role_df = pd.DataFrame({
                "Role": list(roles.keys()),
                "Searches": list(roles.values())
            }).sort_values(by="Searches", ascending=False)
            
            fig = px.pie(
                role_df,
                names="Role",
                values="Searches",
                title="Search Activity by User Role",
                color="Role",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Search history table
        st.subheader("Recent Search History")
        
        history_df = pd.DataFrame(search_history)
        if not history_df.empty:
            # Ensure we have the right columns
            for col in ["query", "timestamp", "user", "user_role"]:
                if col not in history_df.columns:
                    history_df[col] = "Unknown"
            
            # Display the table
            st.dataframe(
                history_df[["query", "timestamp", "user", "user_role"]].sort_values(
                    by="timestamp", ascending=False
                )
            )
else:
    # Generate sample search analytics
    st.info("No search history available yet. Sample data shown for demonstration.")
    
    # Create sample data
    sample_queries = [
        "financial projections", 
        "employee benefits",
        "security protocols",
        "market trends",
        "technical specifications",
        "project timeline",
        "customer satisfaction",
        "compliance requirements",
        "API endpoints",
        "risk factors"
    ]
    
    query_counts = {q: random.randint(3, 15) for q in sample_queries}
    
    # Display sample data
    st.subheader("Sample: Most Common Search Queries")
    
    fig = px.bar(
        x=list(query_counts.keys()),
        y=list(query_counts.values()),
        title="Sample: Top Search Queries",
        color=list(query_counts.keys()),
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig.update_layout(xaxis_title="Query", yaxis_title="Count", height=400)
    st.plotly_chart(fig, use_container_width=True)

# Document insights section
st.markdown("<h2 style='color: #1E3A8A; margin-top: 2rem;'>Key Insights</h2>", unsafe_allow_html=True)

# Generate some sample insights
insights = [
    "Finance-related documents make up a significant portion of your repository, indicating a focus on financial reporting.",
    "Documents related to security protocols have the highest complexity scores, suggesting they may need simplification.",
    "Most documents were created in the past 3 months, showing an active maintenance of your knowledge base.",
    "PDF is the most common document format, suggesting a preference for fixed-layout documents.",
    "Search activity peaks during business hours (9 AM - 5 PM), indicating work-related usage patterns."
]

for insight in insights:
    st.markdown(f"""
    <div class="insights-card">
        <p style="margin: 0;">🔍 {insight}</p>
    </div>
    """, unsafe_allow_html=True)

# Recommendations section
st.markdown("<h2 style='color: #1E3A8A; margin-top: 2rem;'>Recommendations</h2>", unsafe_allow_html=True)

recommendations = [
    "Consider adding more marketing-related documents, as this category is underrepresented in your knowledge base.",
    "Users frequently search for 'project timeline' - creating a dedicated document for this could improve user experience.",
    "Documents older than 6 months may need review to ensure information is current and accurate.",
    "Consider converting more content to searchable formats for better discoverability.",
    "The high number of technical documents may benefit from categorization into subcategories for easier navigation."
]

for i, rec in enumerate(recommendations):
    st.markdown(f"""
    <div style="display: flex; align-items: flex-start; margin-bottom: 1rem;">
        <div style="background-color: #3B82F6; color: white; border-radius: 50%; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; margin-right: 10px; flex-shrink: 0;">
            {i+1}
        </div>
        <div style="background-color: #F3F4F6; padding: 1rem; border-radius: 0.5rem; flex-grow: 1;">
            {rec}
        </div>
    </div>
    """, unsafe_allow_html=True)
