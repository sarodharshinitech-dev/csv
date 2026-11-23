import os
import json
import streamlit as st
from PIL import Image
import pandas as pd
from backend.inference import SafetyInferenceEngine
from dataset.mock_generator import generate_mock_dataset, generate_sample_test_images
from models.train import train_pipeline

# 1. Page Configuration
st.set_page_config(
    page_title="Construction Site Worker Safety AI",
    page_icon="👷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Premium Theme CSS Injection
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Layout Styling */
    .main {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    
    /* Glassmorphic Container */
    .glass-card {
        background: rgba(30, 34, 45, 0.65);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    /* Status Card - Safe */
    .status-safe {
        border-left: 6px solid #2ea043;
        background: rgba(46, 160, 67, 0.1);
        padding: 16px;
        border-radius: 8px;
        margin-top: 10px;
        color: #56d364;
    }
    
    /* Status Card - Unsafe */
    .status-unsafe {
        border-left: 6px solid #f85149;
        background: rgba(248, 81, 73, 0.1);
        padding: 16px;
        border-radius: 8px;
        margin-top: 10px;
        color: #ff7b72;
    }
    
    /* Title styling */
    .main-title {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 5px;
    }
    
    .subtitle {
        color: #9ca3af;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    
    /* Button Hovers */
    div.stButton > button:first-child {
        background-color: #d97706;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #f59e0b;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Define directories
MODEL_DIR = os.path.join('models', 'saved_models')
DATASET_DIR = os.path.join('dataset', 'mock_dataset')
SAMPLE_DIR = 'sample_images'

# Init inference engine
@st.cache_resource
def load_inference_engine():
    return SafetyInferenceEngine(model_dir=MODEL_DIR)

# 3. Sidebar Configuration
st.sidebar.markdown("<h2 style='color: #f59e0b; margin-top: 0;'>⚙️ Settings & Pipeline</h2>", unsafe_allow_html=True)

# Status indicators
model_exists = os.path.exists(os.path.join(MODEL_DIR, 'best_model.keras'))
dataset_exists = os.path.exists(DATASET_DIR)

st.sidebar.markdown("### System Status")
st.sidebar.markdown(f"**Dataset status:** {'Available ✅' if dataset_exists else 'Not Found ❌'}")
st.sidebar.markdown(f"**CNN Model status:** {'Trained & Loaded ✅' if model_exists else 'Not Found ❌'}")

# Threshold slider
confidence_threshold = st.sidebar.slider(
    "Prediction Confidence Threshold",
    min_value=0.5,
    max_value=0.99,
    value=0.75,
    step=0.01,
    help="Minimum probability required to confirm class prediction confidence"
)

# Admin Utilities section
st.sidebar.markdown("---")
st.sidebar.markdown("### Admin Controls")
st.sidebar.info("Use these commands to quickly initialize, train, and test the model if you don't have a dataset.")

if st.sidebar.button("📦 Generate Mock Dataset"):
    with st.spinner("Generating synthetic worker images..."):
        generate_mock_dataset(DATASET_DIR)
        generate_sample_test_images(SAMPLE_DIR)
    st.sidebar.success("Generated mock images in 'dataset/mock_dataset' and 'sample_images'!")
    st.rerun()

epochs = st.sidebar.number_input("Training Epochs", min_value=1, max_value=100, value=10)
batch_size = st.sidebar.selectbox("Batch Size", options=[16, 32, 64], index=1)

if st.sidebar.button("🏋️ Train CNN Model"):
    if not dataset_exists:
        st.sidebar.error("Error: Dataset not found. Click 'Generate Mock Dataset' first.")
    else:
        st.sidebar.markdown("**Training status:** Processing... (Check terminal output for logs)")
        # Progress bar placeholder
        progress_bar = st.sidebar.progress(0.0)
        
        with st.spinner("Training Convolutional Neural Network..."):
            try:
                metrics = train_pipeline(
                    dataset_path=DATASET_DIR,
                    save_dir=MODEL_DIR,
                    epochs=epochs,
                    batch_size=batch_size
                )
                progress_bar.progress(1.0)
                st.sidebar.success(f"Training Complete! Test Acc: {metrics['test_accuracy']:.2%}")
                
                # Clear st.cache_resource so it reloads the newly trained model
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Training failed: {str(e)}")

# Header
st.markdown("<h1 class='main-title'>👷 Construction Site Worker Safety AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Convolutional Neural Network (CNN) real-time safety compliance and PPE detection system.</p>", unsafe_allow_html=True)

# Main layout tabs
tab_detector, tab_analytics = st.tabs(["🔍 Real-time Detection", "📊 Model Performance Analytics"])

# Load engine
engine = load_inference_engine()

# TAB 1: DETECTION
with tab_detector:
    # Check if model exists
    if not model_exists or engine.model is None:
        st.warning("⚠️ **No Trained CNN Model Found.**")
        st.info("""
        Please train the model before running predictions:
        1. Click **📦 Generate Mock Dataset** in the sidebar to create sample images.
        2. Click **🏋️ Train CNN Model** to run the TensorFlow training pipeline.
        
        Alternatively, make sure your trained model file `best_model.keras` and class index mapping `classes.json` are placed in the `models/saved_models/` folder.
        """)
    else:
        col_input, col_result = st.columns([1, 1], gap="large")
        
        with col_input:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("### 📥 Image Input Source")
            
            # Sub-options: upload or sample gallery
            input_mode = st.radio("Choose Input Mode:", ["Upload File", "Select Sample Image"])
            
            selected_image = None
            
            if input_mode == "Upload File":
                uploaded_file = st.file_uploader(
                    "Upload construction site image...",
                    type=["jpg", "jpeg", "png"],
                    help="Limit 200MB per file • JPG, JPEG, PNG"
                )
                if uploaded_file is not None:
                    selected_image = Image.open(uploaded_file)
                    st.image(selected_image, caption="Uploaded Image Preview", use_container_width=True)
            
            else:
                st.markdown("Click on any sample image below to run the detector:")
                # Ensure samples exist
                if not os.path.exists(SAMPLE_DIR) or not os.listdir(SAMPLE_DIR):
                    if st.button("Generate Demo Sample Images"):
                        generate_sample_test_images(SAMPLE_DIR)
                        st.rerun()
                
                if os.path.exists(SAMPLE_DIR) and os.listdir(SAMPLE_DIR):
                    samples = sorted(os.listdir(SAMPLE_DIR))
                    selected_sample = st.selectbox("Select a demo worker image:", samples)
                    
                    sample_path = os.path.join(SAMPLE_DIR, selected_sample)
                    selected_image = Image.open(sample_path)
                    st.image(selected_image, caption=f"Selected Sample: {selected_sample}", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_result:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("### 🖥️ AI Detection Output")
            
            if selected_image is None:
                st.info("Awaiting input image. Please upload a file or select a sample from the gallery on the left.")
            else:
                with st.spinner("Analyzing image features..."):
                    result = engine.predict_image(selected_image)
                    
                if "error" in result:
                    st.error(result["error"])
                else:
                    pred_class = result["predicted_class"]
                    confidence = result["confidence"]
                    probabilities = result["probabilities"]
                    
                    # Safe vs Unsafe rendering based on predicted class name
                    # Standard classes: 'safe' vs 'unsafe'
                    is_safe = "unsafe" not in pred_class.lower() and "non-compliant" not in pred_class.lower()
                    
                    if is_safe:
                        status_html = f"""
                        <div class='status-safe'>
                            <h4>✅ SAFE COMPLIANCE DETECTED</h4>
                            <p>Worker appears to be compliant. Safety Helmet and Reflective Vest detected successfully.</p>
                            <p><strong>Predicted Status:</strong> {pred_class.upper()} ({confidence:.2%})</p>
                        </div>
                        """
                    else:
                        status_html = f"""
                        <div class='status-unsafe'>
                            <h4>⚠️ NON-COMPLIANT WORKER DETECTED</h4>
                            <p>Worker is missing required PPE safety equipment (Helmet or High-Vis Vest)!</p>
                            <p><strong>Predicted Status:</strong> {pred_class.upper()} ({confidence:.2%})</p>
                        </div>
                        """
                    
                    st.markdown(status_html, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Metrics & thresholds check
                    col_metric1, col_metric2 = st.columns(2)
                    col_metric1.metric("Predicted State", pred_class.capitalize())
                    col_metric2.metric("Confidence Score", f"{confidence:.2%}")
                    
                    # Highlight if below confidence threshold
                    if confidence < confidence_threshold:
                        st.warning(f"⚠️ Warning: Confidence score ({confidence:.2%}) is below your set threshold ({confidence_threshold:.2%}). Prediction should be manually inspected.")
                    
                    # Probability Bar Chart
                    st.markdown("#### Probability Distribution")
                    prob_df = pd.DataFrame({
                        "Class": list(probabilities.keys()),
                        "Probability": list(probabilities.values())
                    })
                    st.bar_chart(prob_df.set_index("Class"))
                    
            st.markdown("</div>", unsafe_allow_html=True)


# TAB 2: MODEL ANALYTICS
with tab_analytics:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 📊 CNN Model Performance Dashboard")
    
    metrics_path = os.path.join(MODEL_DIR, 'metrics.json')
    curves_path = os.path.join(MODEL_DIR, 'training_curves.png')
    cm_path = os.path.join(MODEL_DIR, 'confusion_matrix.png')
    report_path = os.path.join(MODEL_DIR, 'classification_report.txt')
    
    if not os.path.exists(metrics_path):
        st.info("No training performance metrics found. Run the training process to generate statistical evaluation logs.")
    else:
        # Load JSON metrics
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
        # Summary metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Test Accuracy Score", f"{metrics['test_accuracy']:.2%}")
        col_m2.metric("Test Categorical Loss", f"{metrics['test_loss']:.4f}")
        col_m3.metric("Training Epochs Concluded", str(metrics['epochs_run']))
        
        st.markdown("---")
        
        # Training and Evaluation Visualizations
        col_curves, col_cm = st.columns(2)
        
        with col_curves:
            st.markdown("#### Loss & Accuracy Progression Curves")
            if os.path.exists(curves_path):
                st.image(curves_path, use_container_width=True)
            else:
                st.warning("Curves image not found.")
                
        with col_cm:
            st.markdown("#### Confusion Matrix (Validation/Test Split)")
            if os.path.exists(cm_path):
                st.image(cm_path, use_container_width=True)
            else:
                st.warning("Confusion matrix image not found.")
                
        st.markdown("---")
        
        # Classification report text
        st.markdown("#### Classification Report Details")
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report_txt = f.read()
            st.code(report_txt, language="text")
        else:
            st.warning("Classification report text file not found.")
            
    st.markdown("</div>", unsafe_allow_html=True)
