import os
from uuid import uuid4
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path
from ultralytics import YOLO
from PIL import Image

# --- Configuration ---
# You can change the YOLO model here. Examples: 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
# Ensure the model chosen is available or will be downloaded by YOLO.
YOLO_MODEL_NAME = os.getenv('YOLO_MODEL_NAME', 'yolov8n.pt')

# Default COCO class names. Modify this list if your model is trained on different classes.
# You can find the class names in the .yaml file associated with your custom YOLO model.
YOLO_MODEL_LABELS = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
# Optional: Set a confidence threshold for predictions
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.3))

# Optional: GPU device index, e.g., 0, 1, 2, 3 or 'cpu'
# Set to 'cpu' if you don't have a GPU or want to force CPU.
GPU_DEVICE = os.getenv('GPU_DEVICE', 'cpu')

class YOLOv8Backend(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(YOLOv8Backend, self).__init__(**kwargs)

        # Initialize model_version based on the model name
        self.model_version = YOLO_MODEL_NAME

        # Load the YOLOv8 model
        # The model is loaded into memory when the class is initialized.
        print(f"Attempting to load YOLO model: {YOLO_MODEL_NAME} on device: {GPU_DEVICE}")
        try:
            self.model = YOLO(YOLO_MODEL_NAME)
            self.model.to(GPU_DEVICE) # Ensure model is on the correct device
            print(f"Successfully loaded YOLO model: {YOLO_MODEL_NAME} on device: {GPU_DEVICE}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None # Set model to None if loading fails

        # Get label names from the model if possible, otherwise use the default list
        # For standard YOLO models, names are often in model.names
        try:
            if hasattr(self.model, 'names') and isinstance(self.model.names, dict):
                 # For models where names is a dict like {0: 'name1', 1: 'name2', ...}
                self.labels = [self.model.names[i] for i in range(len(self.model.names))]
            elif hasattr(self.model, 'names') and isinstance(self.model.names, list):
                 # For models where names is a list
                self.labels = self.model.names
            else:
                self.labels = YOLO_MODEL_LABELS
            print(f"Using labels: {self.labels}")
        except Exception as e:
            print(f"Could not get labels from model, using default. Error: {e}")
            self.labels = YOLO_MODEL_LABELS

        if not self.labels:
            print("Warning: No labels found or loaded. Predictions may not have correct class names.")
            self.labels = [f"label_{i}" for i in range(80)] # Fallback generic labels

    def predict(self, tasks, **kwargs):
        if not self.model:
            print("YOLO Model not loaded. Skipping prediction.")
            return []

        predictions = []
        for task in tasks:
            image_url = task['data']['image']
            try:
                # Get local path to the image
                image_path = get_image_local_path(image_url, project_dir=self.project_dir)

                # Open image to get dimensions
                pil_image = Image.open(image_path)
                original_width, original_height = pil_image.size

                # Perform inference
                results = self.model.predict(image_path, conf=CONFIDENCE_THRESHOLD, device=GPU_DEVICE)

                task_predictions = []
                for result in results: # Iterate through results for the image
                    for i, box in enumerate(result.boxes.xyxyn): # Normalized xyxy
                        # Get confidence and class ID
                        conf = float(result.boxes.conf[i])
                        class_id = int(result.boxes.cls[i])

                        # Get label name
                        label = self.labels[class_id] if class_id < len(self.labels) else f"class_{class_id}"

                        # Convert normalized coordinates to Label Studio format (percentages)
                        x_min_norm, y_min_norm, x_max_norm, y_max_norm = box.tolist()

                        prediction_item = {
                            "id": str(uuid4()),
                            "from_name": "label", # This should match the <RectangleLabels name="label"> in your LS config
                            "to_name": "image",   # This should match the <Image name="image"> in your LS config
                            "type": "rectanglelabels",
                            "original_width": original_width,
                            "original_height": original_height,
                            "image_rotation": 0, # Assuming no rotation for now
                            "value": {
                                "rotation": 0,
                                "x": x_min_norm * 100,
                                "y": y_min_norm * 100,
                                "width": (x_max_norm - x_min_norm) * 100,
                                "height": (y_max_norm - y_min_norm) * 100,
                                "rectanglelabels": [label]
                            },
                            "score": conf
                        }
                        task_predictions.append(prediction_item)

                predictions.append({"result": task_predictions, "model_version": self.model_version})

            except Exception as e:
                print(f"Error processing task {task['id']}: {e}")
                predictions.append({"result": [], "model_version": self.model_version})

        return predictions

    def fit(self, event, data, **kwargs):
        # This YOLOv8 backend is for inference only, so training/fitting is not implemented.
        print("Fit method called, but not implemented for this YOLOv8 inference backend.")
        pass

if __name__ == '__main__':
    # This part is optional and useful for testing the backend script directly
    # To test:
    # 1. Save an image as `test_image.jpg` in the same directory as this script.
    # 2. Run `python yolov8_backend.py`
    # It will print the predictions for `test_image.jpg`.

    print("--- Testing YOLOv8Backend ---")
    # Create a dummy task
    dummy_tasks = [{
        "data": {"image": "https://ultralytics.com/images/bus.jpg"}, # Replace with a local path if needed e.g., "test_image.jpg"
        "id": 1
    }]

    # Initialize and predict
    # You might need to set LABEL_STUDIO_PROJECT_DIR environment variable if get_image_local_path has issues finding files
    # For local testing, if 'LABEL_STUDIO_PROJECT_DIR' is not set, get_image_local_path might download to a cache
    # os.environ['LABEL_STUDIO_PROJECT_DIR'] = '.' # Example: current directory

    backend = YOLOv8Backend()

    # Check if model loaded
    if not backend.model:
        print("Model did not load. Exiting test.")
    else:
        print(f"Labels being used by backend: {backend.labels}")
        print(f"Model version: {backend.model_version}")

        predictions = backend.predict(dummy_tasks)
        import json
        print("--- Predictions ---")
        print(json.dumps(predictions, indent=4))

        # Example of how to access specific parts of the prediction:
        if predictions and predictions[0]['result']:
            first_prediction_details = predictions[0]['result'][0]['value']
            print(f"""
Details of the first detected object in the first task:""")
            print(f"  Label: {first_prediction_details['rectanglelabels']}")
            print(f"  X: {first_prediction_details['x']:.2f}%")
            print(f"  Y: {first_prediction_details['y']:.2f}%")
            print(f"  Width: {first_prediction_details['width']:.2f}%")
            print(f"  Height: {first_prediction_details['height']:.2f}%")
            print(f"  Confidence: {predictions[0]['result'][0]['score']:.2f}")
        else:
            print("""
No detections or error in prediction.""")

    print("--- Test Complete ---")
