import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
import detection.detect as detect

# Page configuration
st.set_page_config(
    page_title="YOLOv8 Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced design
st.markdown("""
<style>
    /* Main container styling */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        width: 350px;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* About section styling */
    .about-container {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    
    .about-container h2 {
        color: #000000 !important;
    }
    
    .about-container p {
        color: #000000 !important;
    }
    
    .about-container ul {
        color: #000000 !important;
    }
    
    .about-container li {
        color: #000000 !important;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .feature-card h3 {
        color: #000000 !important;
        margin-top: 0;
    }
    
    .feature-card p {
        color: #000000 !important;
    }
    
    /* Detection section styling */
    .detection-container {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Sidebar text color */
    .sidebar-text {
        color: white !important;
        font-weight: bold;
    }
    
    /* Image container */
    .image-container {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        background-color: #f8f9fa;
    }
    
    /* Metrics styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    
    /* Parameter section styling */
    .param-section {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def train_detection_model():
    """Train the detection model"""
    detect.train()
    st.success("✅ Detection model training completed!")

def main():
    # Sidebar configuration
    with st.sidebar:
        st.markdown('<p class="sidebar-text">🎯 YOLOv8 Detection</p>', unsafe_allow_html=True)
        st.markdown("---")
        
        # App mode selection
        app_mode = st.selectbox(
            '🚀 Choose Mode',
            ['📖 About App', '🔍 Object Detection'],
            help="Select the application mode"
        )
        
        if app_mode == '🔍 Object Detection':
            st.markdown("---")
            st.markdown('<p class="sidebar-text">⚙️ Detection Parameters</p>', unsafe_allow_html=True)
            
            # Primary detection parameters
            st.markdown('<div class="param-section">', unsafe_allow_html=True)
            st.markdown("**🎯 Core Detection Settings**")
            
            # Confidence threshold slider
            confidence = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.35,
                step=0.05,
                help="Minimum confidence score for object detection"
            )
            
            # IoU threshold slider
            iou_threshold = st.slider(
                "IoU Threshold (NMS)",
                min_value=0.0,
                max_value=1.0,
                value=0.45,
                step=0.05,
                help="IoU threshold for Non-Maximum Suppression"
            )
            
            # Maximum detections
            max_det = st.number_input(
                "Max Detections",
                min_value=1,
                max_value=1000,
                value=300,
                step=10,
                help="Maximum number of detections per image"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Advanced parameters
            st.markdown('<div class="param-section">', unsafe_allow_html=True)
            st.markdown("**🔧 Advanced Settings**")
            
            # Image size
            img_size = st.selectbox(
                "Image Size",
                [320, 416, 512, 640, 832, 1024],
                index=3,
                help="Input image size for inference"
            )
            
            # Device selection
            device = st.selectbox(
                "Device",
                ["auto", "cpu", "cuda:0"],
                index=0,
                help="Device for inference (auto, cpu, or cuda:0)"
            )
            
            # Half precision
            half_precision = st.checkbox(
                "Half Precision (FP16)",
                value=False,
                help="Use half precision for faster inference (requires CUDA)"
            )
            
            # Augmentation during inference
            augment = st.checkbox(
                "Test Time Augmentation",
                value=False,
                help="Apply augmentation during inference for better accuracy"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Visualization parameters
            st.markdown('<div class="param-section">', unsafe_allow_html=True)
            st.markdown("**🎨 Visualization Settings**")
            
            # Show confidence scores
            show_conf = st.checkbox(
                "Show Confidence Scores",
                value=True,
                help="Display confidence scores on bounding boxes"
            )
            
            # Show class labels
            show_labels = st.checkbox(
                "Show Class Labels",
                value=True,
                help="Display class labels on bounding boxes"
            )
            
            # Line thickness
            line_thickness = st.slider(
                "Box Line Thickness",
                min_value=1,
                max_value=10,
                value=3,
                help="Thickness of bounding box lines"
            )
            
            # Font size
            font_size = st.slider(
                "Font Size",
                min_value=0.3,
                max_value=2.0,
                value=0.8,
                step=0.1,
                help="Font size for labels and confidence scores"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Class filtering
            st.markdown('<div class="param-section">', unsafe_allow_html=True)
            st.markdown("**🏷️ Class Filtering**")
            
            # Enable class filtering
            enable_class_filter = st.checkbox(
                "Filter Specific Classes",
                value=False,
                help="Only detect specific object classes"
            )
            
            if enable_class_filter:
                # Common COCO classes for example
                class_options = [
                    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
                    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
                    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee"
                ]
                
                selected_classes = st.multiselect(
                    "Select Classes to Detect",
                    class_options,
                    default=[],
                    help="Choose specific classes to detect (leave empty for all classes)"
                )
            else:
                selected_classes = []
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Image upload
            st.markdown("---")
            st.markdown('<p class="sidebar-text">📁 Upload Image</p>', unsafe_allow_html=True)
            img_file_buffer = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png'],
                help="Upload an image for object detection"
            )
            
            # Training option
            st.markdown("---")
            st.markdown('<p class="sidebar-text">🔧 Model Training</p>', unsafe_allow_html=True)
            if st.button("🚀 Train Detection Model", help="Retrain the detection model"):
                with st.spinner("Training in progress..."):
                    train_detection_model()

    # Main content area
    if app_mode == '📖 About App':
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>🎯 YOLOv8 Object Detection</h1>
            <p>Advanced Real-time Object Detection with Deep Learning</p>
        </div>
        """, unsafe_allow_html=True)
        
        # About content
        st.markdown("""
        <div class="about-container">
            <h2>🚀 Welcome to YOLOv8 Object Detection</h2>
            <p style="font-size: 18px; line-height: 1.6; color: #000000;">
                Experience the power of state-of-the-art object detection with YOLOv8! 
                Our application leverages the latest YOLO (You Only Look Once) algorithm 
                to detect objects in images with remarkable speed and accuracy.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Features section
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h3>⚡ Lightning Fast</h3>
                <p>YOLOv8 processes images in real-time, detecting multiple objects in a single pass through the neural network.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h3>🎯 High Accuracy</h3>
                <p>Advanced anchor-free detection head and improved loss function deliver superior detection performance.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h3>🔧 Easy to Use</h3>
                <p>Simple interface with adjustable confidence thresholds and support for multiple image formats.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h3>🧠 Smart Detection</h3>
                <p>Powered by advanced backbone network for robust feature extraction and object recognition.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Key improvements section
        st.markdown("""
        <div class="about-container">
            <h2>🌟 YOLOv8 Key Improvements</h2>
            <ul style="font-size: 16px; line-height: 1.8; color: #000000;">
                <li><strong>🔗 New Backbone Network:</strong> Enhanced architecture for better feature extraction</li>
                <li><strong>🎯 Anchor-Free Detection:</strong> Direct center prediction instead of anchor box offsets</li>
                <li><strong>📈 Improved Loss Function:</strong> Optimized training process for better convergence</li>
                <li><strong>⚡ Real-time Performance:</strong> Optimized for speed without compromising accuracy</li>
                <li><strong>🎛️ Comprehensive Parameters:</strong> Full control over detection settings and visualization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Getting started
        st.markdown("""
        <div class="about-container">
            <h2>🚀 Get Started</h2>
            <p style="font-size: 16px; line-height: 1.6; color: #000000;">
                Ready to detect objects in your images? Switch to <strong>Object Detection</strong> mode 
                from the sidebar, upload your image, adjust the various parameters including confidence threshold,
                IoU threshold, visualization settings, and more. Watch YOLOv8 work its magic! 🎆
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    elif app_mode == '🔍 Object Detection':
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>🔍 Object Detection</h1>
            <p>Upload an image and detect objects with YOLOv8</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detection interface
        st.markdown('<div class="detection-container">', unsafe_allow_html=True)
        
        # Default demo image
        DEMO_IMAGE = "DEMO_IMAGES/BloodImage_00000_jpg.rf.5fb00ac1228969a39cee7cd6678ee704.jpg"
        
        # Image processing
        if img_file_buffer is not None:
            # User uploaded image
            img = cv.imdecode(np.frombuffer(img_file_buffer.read(), np.uint8), 1)
            image = np.array(Image.open(img_file_buffer))
            st.success("✅ Image uploaded successfully!")
        else:
            # Demo image
            try:
                img = cv.imread(DEMO_IMAGE)
                image = np.array(Image.open(DEMO_IMAGE))
                st.info("ℹ️ Using demo image. Upload your own image from the sidebar.")
            except:
                st.error("❌ Demo image not found. Please upload an image.")
                st.stop()
        
        # Display original image in sidebar
        with st.sidebar:
            st.markdown("---")
            st.markdown('<p class="sidebar-text">🖼️ Original Image</p>', unsafe_allow_html=True)
            st.image(image, use_column_width=True)
        
        # Detection metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Confidence</h3>
                <h2>{confidence:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔗 IoU Threshold</h3>
                <h2>{iou_threshold:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📐 Image Size</h3>
                <h2>{img_size}px</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Max Detections</h3>
                <h2>{max_det}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Run detection
        st.subheader("🎯 Detection Results")
        
        # Create parameters dictionary
        detection_params = {
            'confidence': confidence,
            'iou_threshold': iou_threshold,
            'max_det': max_det,
            'img_size': img_size,
            'device': device,
            'half_precision': half_precision,
            'augment': augment,
            'show_conf': show_conf,
            'show_labels': show_labels,
            'line_thickness': line_thickness,
            'font_size': font_size,
            'selected_classes': selected_classes if enable_class_filter else None
        }
        
        try:
            with st.spinner("🔍 Detecting objects..."):
                # Run detection with all parameters
                detect.predict(img, detection_params, st)
        except Exception as e:
            st.error(f"❌ Detection failed: {str(e)}")
            st.info("💡 Make sure the detection module is properly configured.")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass