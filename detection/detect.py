from ultralytics import YOLO
import os
import cv2 as cv
import numpy as np
from PIL import Image

def train():
    model = YOLO('yolov8n.yaml')  # build a new model from scratch
    model.train(data="../data/detection/data.yaml", epochs=100)  # train the model
    

def predict(img, params, st):
    """
    Enhanced prediction function with multiple parameters
    
    Args:
        img: Input image (OpenCV format)
        params: Dictionary containing all detection parameters
        st: Streamlit object for display
    """
    # Detection model
    model_path = os.path.join('.', 'detection', 'yolo_experiments', 'yolov8s_training', 'weights', 'best.pt')
    model = YOLO(model_path)
    
    # Extract parameters with defaults
    confidence = params.get('confidence', 0.35)
    iou_threshold = params.get('iou_threshold', 0.45)
    max_det = params.get('max_det', 300)
    img_size = params.get('img_size', 640)
    device = params.get('device', 'auto')
    half_precision = params.get('half_precision', False)
    augment = params.get('augment', False)
    show_conf = params.get('show_conf', True)
    show_labels = params.get('show_labels', True)
    line_thickness = params.get('line_thickness', 3)
    font_size = params.get('font_size', 0.8)
    selected_classes = params.get('selected_classes', None)
    
    # Prepare prediction arguments
    predict_args = {
        'source': img,
        'conf': confidence,
        'iou': iou_threshold,
        'max_det': max_det,
        'imgsz': img_size,
        'device': device,
        'half': half_precision,
        'augment': augment,
        'verbose': False
    }
    
    # Add class filtering if specified
    if selected_classes and len(selected_classes) > 0:
        # Convert class names to class indices
        # This assumes you know the class mapping or can get it from the model
        try:
            class_indices = []
            for class_name in selected_classes:
                # Try to find the class index from model names
                if hasattr(model, 'names') and model.names:
                    for idx, name in model.names.items():
                        if name.lower() == class_name.lower():
                            class_indices.append(idx)
                            break
            
            if class_indices:
                predict_args['classes'] = class_indices
        except Exception as e:
            st.warning(f"⚠️ Could not apply class filtering: {str(e)}")
    
    # Run prediction
    try:
        results = model.predict(**predict_args)
        result = results[0]
        
        # Print detection info
        num_objects = len(result.boxes) if result.boxes is not None else 0
        print(f"\n[INFO] Number of objects detected: {num_objects}")
        
        # Display detection statistics
        if num_objects > 0:
            st.success(f"✅ Detected {num_objects} objects!")
            
            # Show detection details
            if result.boxes is not None:
                classes_detected = {}
                confidences = []
                
                for box in result.boxes:
                    if hasattr(box, 'cls') and hasattr(box, 'conf'):
                        class_id = int(box.cls.item())
                        confidence_score = float(box.conf.item())
                        class_name = model.names.get(class_id, f"Class_{class_id}")
                        
                        if class_name not in classes_detected:
                            classes_detected[class_name] = 0
                        classes_detected[class_name] += 1
                        confidences.append(confidence_score)
                
                # Display class statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🏷️ Unique Classes", len(classes_detected))
                
                with col2:
                    if confidences:
                        avg_conf = sum(confidences) / len(confidences)
                        st.metric("📊 Avg Confidence", f"{avg_conf:.3f}")
                
                with col3:
                    if confidences:
                        max_conf = max(confidences)
                        st.metric("🎯 Max Confidence", f"{max_conf:.3f}")
                
                # Show class breakdown
                if classes_detected:
                    st.write("**🏷️ Detected Classes:**")
                    class_text = ", ".join([f"{name} ({count})" for name, count in classes_detected.items()])
                    st.write(class_text)
        else:
            st.warning("⚠️ No objects detected. Try adjusting the confidence threshold.")
        
        # Generate annotated image
        annotated_img = None
        for r in results:
            # Plot with custom parameters
            im_array = r.plot(
                conf=show_conf,
                labels=show_labels,
                line_width=line_thickness,
                font_size=font_size
            )
            # Convert BGR to RGB for display
            annotated_img = Image.fromarray(im_array[..., ::-1])
        
        # Display the result
        if annotated_img:
            st.subheader('🖼️ Detection Results')
            st.image(annotated_img, use_container_width=True, caption="YOLOv8 Detection Results")
        else:
            st.error("❌ Failed to generate annotated image")
        
        # Optional: Display detection details in expandable section
        if result.boxes is not None and len(result.boxes) > 0:
            with st.expander("📋 Detailed Detection Information"):
                for i, box in enumerate(result.boxes):
                    if hasattr(box, 'xyxy') and hasattr(box, 'conf') and hasattr(box, 'cls'):
                        # Extract box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence_score = float(box.conf.item())
                        class_id = int(box.cls.item())
                        class_name = model.names.get(class_id, f"Class_{class_id}")
                        
                        # Calculate box area and center
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        st.write(f"""
                        **Detection {i+1}:**
                        - 🏷️ **Class:** {class_name}
                        - 🎯 **Confidence:** {confidence_score:.3f}
                        - 📐 **Bounding Box:** ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})
                        - 📏 **Size:** {width:.1f} × {height:.1f} (Area: {area:.0f})
                        - 🎪 **Center:** ({center_x:.1f}, {center_y:.1f})
                        """)
        
        # Save results option
        if st.button("💾 Save Detection Results"):
            try:
                if annotated_img:
                    # Save the annotated image
                    save_path = "detection_results.jpg"
                    annotated_img.save(save_path)
                    st.success(f"✅ Results saved as {save_path}")
                else:
                    st.error("❌ No results to save")
            except Exception as e:
                st.error(f"❌ Failed to save results: {str(e)}")
    
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        print(f"[ERROR] Prediction error: {str(e)}")
        
        # Fallback: try with minimal parameters
        try:
            st.info("🔄 Trying with basic parameters...")
            basic_results = model.predict(img, conf=confidence)
            if basic_results and len(basic_results) > 0:
                basic_result = basic_results[0]
                im_array = basic_result.plot()
                im = Image.fromarray(im_array[..., ::-1])
                st.image(im, use_container_width=True, caption="Basic Detection Results")
            else:
                st.error("❌ Basic prediction also failed")
        except Exception as basic_error:
            st.error(f"❌ Basic prediction failed: {str(basic_error)}")


def get_model_info(model_path=None):
    """
    Get information about the loaded model
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dictionary with model information
    """
    try:
        if model_path is None:
            model_path = os.path.join('.', 'detection', 'yolo_experiments', 'yolov8s_training', 'weights', 'best.pt')
        
        model = YOLO(model_path)
        
        info = {
            'model_path': model_path,
            'model_exists': os.path.exists(model_path),
            'class_names': getattr(model, 'names', {}),
            'num_classes': len(getattr(model, 'names', {}))
        }
        
        return info
    except Exception as e:
        return {
            'error': str(e),
            'model_path': model_path,
            'model_exists': False
        }