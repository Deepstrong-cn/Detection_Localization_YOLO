import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # Paths (modify as needed)
    WEIGHTS = 'runs/train/weights/best.pt'  # Trained model weights (from training)
    SOURCE = 'DATA/dataset/images'          # Source images or videos
    
    # Load model
    model = YOLO(WEIGHTS)

    # Run inference (adjust parameters as needed)
    model.predict(
        source=SOURCE,
        imgsz=640,                        # Input image size
        conf=0.25,                        # Confidence threshold
        iou=0.45,                         # IoU threshold for NMS
        save=True,                        # Save results
        save_txt=False,                   # Save labels as .txt
        save_crop=False,                  # Save cropped images
        line_width=1,                     # Bounding box line width
        show_conf=True,                   # Show confidence scores
        show_labels=True,                 # Show labels
        project='runs/predict',           # Save directory
    )