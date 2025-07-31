import streamlit as st
from document_processor import DocumentProcessor
import os

def main():
    st.title("Document Processing Test")
    
    # Process documents from the documents folder
    with st.spinner("Processing documents..."):
        processor = DocumentProcessor()
        docs = processor.process_directory('documents')
        
        st.success(f"Successfully processed {len(docs)} document chunks")
        
        # Display document summary
        st.subheader("Document Summary")
        
        # Total document chunks
        st.write(f"Number of document chunks: {len(docs)}")
        
        # Count unique document files
        unique_files = set()
        doc_files_dict = {}
        
        for doc in docs:
            if "source" in doc.metadata:
                file_path = doc.metadata["source"]
                file_name = os.path.basename(file_path)
                unique_files.add(file_path)
                
                # Count documents by file
                if file_name in doc_files_dict:
                    doc_files_dict[file_name] += 1
                else:
                    doc_files_dict[file_name] = 1
        
        # Display unique document count and names
        st.write(f"Unique document files: {len(unique_files)}")
        
        # Display document list with chunks per file
        if doc_files_dict:
            st.subheader("Document Details")
            for doc_name, count in doc_files_dict.items():
                st.markdown(f"📄 **{doc_name}** - {count} chunks")
        
        # Show the content of the first few documents
        st.subheader("Sample Document Content")
        for i, doc in enumerate(docs[:5]):
            with st.expander(f"Document Chunk {i+1} - Source: {os.path.basename(doc.metadata.get('source', 'Unknown'))}"):
                st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)

if __name__ == "__main__":
    main()
