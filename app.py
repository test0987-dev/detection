import os
import cv2
import torch
import numpy as np
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from huggingface_hub import hf_hub_download
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, Instances
import logging
import uuid
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'static/results'
MODEL_FOLDER = 'model'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Class names in EXACT training order
CLASS_NAMES = [
    "Antelope",  # class 0
    "Lion",      # class 1
    "elephant",  # class 2
    "zebra",     # class 3
    "Gorilla",   # class 4
    "Wolf",      # class 5
    "Leopard",   # class 6
    "Giraffe"    # class 7
]

# Create directories
for folder in [UPLOAD_FOLDER, RESULTS_FOLDER, MODEL_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Global model variable
predictor = None

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def setup_model():
    """Initialize model with strict detection settings"""
    global predictor
    
    try:
        cfg = get_cfg()
        
        # Base configuration with stricter thresholds
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85  
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000 
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 500
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5  
        
        # Load weights
        model_file = os.path.join(MODEL_FOLDER, "model_final.pth")
        if not os.path.exists(model_file):
            hf_hub_download(
                repo_id="sandbox338/wildlife-detector-detectron2",
                filename="model_final.pth",
                local_dir=MODEL_FOLDER,
                local_dir_use_symlinks=False
            )
        cfg.MODEL.WEIGHTS = model_file
        
        # Device configuration
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Register metadata
        MetadataCatalog.get("wildlife").thing_classes = CLASS_NAMES
        
        # Create predictor
        predictor = DefaultPredictor(cfg)
        
        return predictor
        
    except Exception as e:
        logger.error(f"Model setup failed: {str(e)}")
        raise

def filter_detections(instances, image):
    """Apply strict filtering to eliminate false positives"""
    filtered = []
    boxes = instances.pred_boxes.tensor.cpu().numpy()
    scores = instances.scores.cpu().numpy()
    classes = instances.pred_classes.cpu().numpy()
    
    # Get the detection with highest confidence
    if len(scores) > 0:
        max_idx = np.argmax(scores)
        filtered.append((boxes[max_idx], scores[max_idx], classes[max_idx]))
    
    # For multiple animals in same image, add others with >70% confidence
    for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
        if i != max_idx and score > 0.7:
            # Check for significant overlap with existing detections
            valid = True
            for existing_box, _, _ in filtered:
                iou = calculate_iou(box, existing_box)
                if iou > 0.3:  # If overlapping too much, skip
                    valid = False
                    break
            if valid:
                filtered.append((box, score, cls))
    
    return filtered

def calculate_iou(box1, box2):
    """Calculate Intersection over Union for two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return inter_area / float(box1_area + box2_area - inter_area)

def process_image(image_path):
    """Process image with strict filtering"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Invalid image file")
        
        outputs = predictor(image)
        instances = outputs["instances"]
        
        # Apply strict filtering
        filtered = filter_detections(instances, image)
        
        # Prepare detections
        detections = []
        for box, score, cls in filtered:
            detections.append({
                'class': CLASS_NAMES[cls],
                'confidence': float(score),
                'bbox': box.tolist(),
                'bbox_normalized': [
                    float(box[0]/image.shape[1]),
                    float(box[1]/image.shape[0]),
                    float(box[2]/image.shape[1]),
                    float(box[3]/image.shape[0])
                ]
            })
        
        # Create visualization
        metadata = MetadataCatalog.get("wildlife")
        visualizer = Visualizer(image[:, :, ::-1], metadata=metadata, scale=0.8)
        
        if detections:
            # Create fake instances for visualization
            boxes = [det['bbox'] for det in detections]
            scores = [det['confidence'] for det in detections]
            classes = [CLASS_NAMES.index(det['class']) for det in detections]
            
            vis_instances = Instances(
                image_size=image.shape[:2],
                pred_boxes=Boxes(torch.tensor(boxes)),
                scores=torch.tensor(scores),
                pred_classes=torch.tensor(classes)
            )
            
            vis_output = visualizer.draw_instance_predictions(vis_instances)
            result_image = vis_output.get_image()[:, :, ::-1]
        else:
            result_image = image.copy()
            cv2.putText(result_image, "No confident detections", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return result_image, detections
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html', classes=CLASS_NAMES)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only JPG, JPEG, PNG allowed.'}), 400
            
        # Save file
        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Process image
        result_image, detections = process_image(filepath)
        
        # Save result
        result_filename = f"result_{filename}"
        result_path = os.path.join(RESULTS_FOLDER, result_filename)
        cv2.imwrite(result_path, result_image)
        
        # Prepare response
        response = {
            'success': True,
            'result_image': f'/static/results/{result_filename}',
            'detections': detections,
            'detection_count': len(detections),
            'confidence_threshold': 0.7
        }
        
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass
            
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'classes': CLASS_NAMES,
        'confidence_threshold': 0.3
    })

if __name__ == '__main__':
    try:
        logger.info("Starting wildlife detection service...")
        setup_model()
        app.run(host='0.0.0.0', port=8000, debug=True)
    except Exception as e:
        logger.error(f"Failed to start: {str(e)}")