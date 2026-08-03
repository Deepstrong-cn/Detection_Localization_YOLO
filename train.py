from ultralytics import YOLO
import os

if __name__ == '__main__':
    # GPU configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # Paths (modify as needed)
    MODEL_CONFIG = 'ultralytics/cfg/models/v8/yolov8-p2-spd-GAM-ghost.yaml'  # Model config file
    PRETRAINED = 'weights/yolov8n.pt'                                         # Pretrained weights
    DATA_CONFIG = 'DATA/yolov8.yaml'                                          # Dataset config file

    # Load model and pretrained weights
    model = YOLO(MODEL_CONFIG)
    model.load(PRETRAINED)

    model.train(
        data=DATA_CONFIG,
        cache=False,
        imgsz=640,
        epochs=400,
        batch=84,
        workers=32,
        device=[0, 1],
        resume=False,
        project='runs/detect',
        exist_ok=False,
    )