import streamlit as st
import sys
import os
from model_manager import ModelManager

def check_model_availability():
    """
    Check model availability at startup and display appropriate messages
    """
    try:
        st.write("### Checking Model Availability")
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Update status
        status_text.text("Initializing model manager...")
        progress_bar.progress(20)
        
        # Initialize model manager
        model_manager = ModelManager()
        
        # Update status
        status_text.text("Checking API key configuration...")
        progress_bar.progress(40)
        
        # Check if API key is configured
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            status_text.text("Google API key not found. Will use local models.")
            progress_bar.progress(60)
        else:
            status_text.text("Google API key found. Checking connectivity...")
            progress_bar.progress(60)
        
        # Try to get model
        status_text.text("Loading model...")
        progress_bar.progress(80)
        
        _, model_info = model_manager.get_model()
        
        # Update status
        status_text.text(f"Model loaded: {model_info.get('name')} ({model_info.get('type')})")
        progress_bar.progress(100)
        
        # Display result
        if model_info.get('type') == 'API':
            st.success(f"Successfully connected to {model_info.get('name')} {model_info.get('version')} API")
        else:
            # Check if device selection was forced through UI
            if os.getenv("FORCE_CPU", "false").lower() == "true":
                st.info(f"Using local model on CPU (manually selected): {model_info.get('name')}")
            elif os.getenv("FORCE_GPU", "false").lower() == "true":
                if model_info.get('device') == 'cuda':
                    st.success(f"Using local model on GPU (manually selected): {model_info.get('name')}")
                else:
                    st.warning(f"Requested GPU but using CPU - GPU may not be available: {model_info.get('name')}")
            elif os.getenv("FORCE_LOCAL_MODEL", "false").lower() == "true":
                st.info(f"Using local model by configuration: {model_info.get('name')} on {model_info.get('device', 'Unknown')}")
            else:
                st.warning(f"Using local fallback model: {model_info.get('name')} on {model_info.get('device', 'Unknown')}")
                if 'device' in model_info and model_info['device'] == 'cpu':
                    st.info("⚠️ Running on CPU. Performance might be limited. Try selecting GPU in settings if available.")
        
        return model_info
    
    except Exception as e:
        st.error(f"Error checking model availability: {str(e)}")
        return {"name": "Unknown", "type": "Error", "error": str(e)}

if __name__ == "__main__":
    # This can be run as a standalone script to check model availability
    import streamlit.web.cli as stcli
    
    sys.argv = ["streamlit", "run", __file__]
    sys.exit(stcli.main())
