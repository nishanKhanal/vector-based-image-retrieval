import streamlit as st
import torch
import os
import json
import numpy as np
from PIL import Image
import requests
from pathlib import Path

# Import our modules
from src.model import ResNetTransferModel
from utils import (
    preprocess_image, extract_features,
    build_faiss_index, load_faiss_index, save_faiss_index, search_similar_images,
    display_results
)

# Set page configuration
st.set_page_config(
    page_title="Image Retrieval Demo", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# File paths (default)
MODEL_DIR = "weights"
DATA_DIR = "data"
DEFAULT_MODEL = "resnet18"
MODEL_PATH = os.path.join(MODEL_DIR, f"model-{DEFAULT_MODEL}.pth")
DEFAULT_FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index-{DEFAULT_MODEL}.bin")
DEFAULT_FEATURES_PATHS_FILE = os.path.join(DATA_DIR, "features_paths.json")


def get_model_url(model_name):
    return f"https://huggingface.co/nishan98/image-retrieval/resolve/main/model-{model_name}.pth"

@st.cache_resource
def download_model_if_needed(model_name):
    model_path = Path("weights").resolve() / f"model-{model_name}.pth"
    if not model_path.exists():
        st.warning(f"Model file for {model_name} not found. Downloading...")
        url = get_model_url(model_name)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(4*1024*1024):
                f.write(chunk)
        st.success(f"Downloaded model for {model_name}")
    return model_path

@st.cache_resource
def load_model(model_path, model_name, num_classes=101):
    """Load the trained model"""
    # Determine device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.sidebar.info(f"Using device: {device}")
    
    model = ResNetTransferModel(num_classes=num_classes, embedding_size=128, base_model = model_name, pretrained=False).to(device)
    
    try:
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            st.success("Model loaded successfully!")
        else:
            st.error(f"Model not found at {model_path}. Please run precompute.sh first.")
            return None, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, device
    
    model.eval()
    return model, device

def get_available_models(available_models_info_path="weights/available_models.json"):
    """Get a list of available models"""

    with open(available_models_info_path, 'r') as f:
        model_names = json.load(f)
        return model_names

def main():
    # Set up header
    st.markdown("<h1 style='text-align: center;'>AI Powered Image Retrieval Demo</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Find Similar Images using Vector Databases</p>", unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.title("Settings")
    
    # Model selection
    st.sidebar.subheader("Model Selection")

    # Get available models
    available_models = get_available_models()

    if available_models:
        selected_model = st.sidebar.selectbox("Select a model", available_models)
        
        # Auto-update paths based on selected model
        faiss_index_path = os.path.join(DATA_DIR, f"faiss_index-{selected_model}.bin")

        # Display auto-filled, read-only paths
        st.sidebar.text_input("FAISS Index Path", value=faiss_index_path, disabled=True)
    else:
        st.sidebar.warning("No available models found in the weights directory.")
        selected_model = None
        faiss_index_path = None

    features_paths_file = DEFAULT_FEATURES_PATHS_FILE
    st.sidebar.text_input("Features Paths File", value=features_paths_file, disabled=True)
        
    # Settings
    num_results = st.sidebar.slider("Number of Results", min_value=1, max_value=10, value=5)
    show_all_categories = st.sidebar.checkbox("Show All Categories", value=False)
    
    # Upload image section
    st.header("Upload an Image")
    
    # Create columns for upload and buttons

    # Row 1: Upload Image
    upload_col = st.columns(1)[0]  # Single full-width column
    uploaded_file = upload_col.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Row 2: Search and Clear Buttons
    button_col1, button_col2 = st.columns(2)

    with button_col1:
        search_button = st.button("Search")

    with button_col2:
        clear_button = st.button("Clear")
    
    # Initialize session state for results
    if 'results_displayed' not in st.session_state:
        st.session_state.results_displayed = False
        
    # Display all categories if requested
    if show_all_categories and os.path.exists(features_paths_file):
        try:
            with open(features_paths_file, 'r') as f:
                indexed_paths = json.load(f)
                
            # Extract all unique categories
            categories = set()
            for item in indexed_paths:
                if "category" in item:
                    categories.add(item["category"])
            
            # Display categories
            st.sidebar.subheader("Available Categories")
            
            # Create a grid display for categories
            category_list = sorted(list(categories))
            rows = [category_list[i:i+2] for i in range(0, len(category_list), 2)]
            
            for row in rows:
                cols = st.sidebar.columns(2)
                for i, category in enumerate(row):
                    cols[i].write(f"• {category}")
        except Exception as e:
            st.sidebar.warning(f"Could not load categories: {e}")
    
    # Clear button logic
    if clear_button:
        # Reset all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Only load model and data if needed
    if uploaded_file is not None or search_button:
        # Load model and required data
        if selected_model:
            model_path = os.path.join(MODEL_DIR, f"model-{selected_model}.pth")
        else:
            model_path = MODEL_PATH  # fallback to default

        model_path = download_model_if_needed(selected_model or DEFAULT_MODEL)
        model, device = load_model(model_path, selected_model)
        if not model:
            st.error("Model not found. Please run precompute.sh first.")
            st.stop()
        
        # Try to load FAISS index
        faiss_index = load_faiss_index(faiss_index_path)
        if faiss_index is None:
            st.error(f"FAISS index not found at {faiss_index_path}. Please run precompute.sh first.")
            st.stop()
            
        # Try to load paths with metadata
        if os.path.exists(features_paths_file):
            try:
                with open(features_paths_file, 'r') as f:
                    indexed_paths = json.load(f)
                st.sidebar.info(f"FAISS index contains {len(indexed_paths)} images")
            except Exception as e:
                st.warning(f"Error loading features paths: {e}")
                indexed_paths = None
        else:
            st.warning(f"Features paths file not found at {features_paths_file}. Results may not be accurate.")
            indexed_paths = None
        
        # Check if we have paths data
        if not indexed_paths:
            st.error("Required files not found. Please run precompute.sh first.")
            st.stop()
        
        # Only perform search when button is clicked AND there's an uploaded file
        if search_button and uploaded_file:
            # Process image and display results
            query_image = Image.open(uploaded_file).convert('RGB')
            
            # Display the uploaded image in the sidebar
            st.sidebar.header("Uploaded Image")
            st.sidebar.image(query_image, use_container_width=True)
            
            # Process the query image
            with st.spinner("Processing image..."):
                # Process the query image with the model
                query_tensor = preprocess_image(query_image, device)
                query_feature = extract_features(model, query_tensor, device)
                
                # Search for similar images
                similarities, indices = search_similar_images(query_feature, faiss_index, k=num_results)
                
                # Display results
                display_results(similarities, indices, indexed_paths)
                
                # Mark that we've displayed results
                st.session_state.results_displayed = True
        elif search_button and not uploaded_file:
            st.warning("Please upload an image first before searching.")
    
    # Footer
    st.markdown("---")
    

if __name__ == "__main__":
    main()