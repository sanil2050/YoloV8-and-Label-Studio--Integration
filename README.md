# Setting up Label Studio with YOLOv8 for Automatic Pre-annotation (Local, No Docker)

This guide provides a comprehensive, step-by-step approach to installing and configuring Label Studio and YOLOv8 on your local machine for automatic pre-annotation of images. This setup does not require Docker.

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Label Studio Installation](#label-studio-installation)
4. [YOLOv8 Installation](#yolov8-installation)
5. [Label Studio ML Backend Setup](#label-studio-ml-backend-setup)
    - [Installing the ML Backend](#installing-the-ml-backend)
    - [Creating the YOLOv8 Backend Script](#creating-the-yolov8-backend-script)
6. [Configuring YOLOv8 in Label Studio](#configuring-yolov8-in-label-studio)
    - [Launching the ML Backend](#launching-the-ml-backend)
    - [Connecting Label Studio to the ML Backend](#connecting-label-studio-to-the-ml-backend)
    - [Setting up the Labeling Interface](#setting-up-the-labeling-interface)
7. [Usage Guide](#usage-guide)
    - [Creating a Project](#creating-a-project)
    - [Importing Data](#importing-data)
    - [Automatic Pre-annotation](#automatic-pre-annotation)
    - [Reviewing and Correcting Annotations](#reviewing-and-correcting-annotations)
8. [Troubleshooting](#troubleshooting)
9. [Conclusion](#conclusion)

## 1. Introduction
(Brief overview of Label Studio, YOLOv8, and the benefits of pre-annotation)

## 2. Prerequisites

Before you begin, ensure you have the following installed on your local machine:

*   **Python**: Version 3.8 or higher. You can download it from [python.org](https://www.python.org/downloads/).
    *   Verify your Python installation by opening a terminal or command prompt and typing:
        ```bash
        python --version
        # or
        python3 --version
        ```
*   **pip (Python Package Installer)**: Usually comes with Python. If not, or if you need to upgrade, follow the instructions on [pip.pypa.io](https://pip.pypa.io/en/stable/installation/).
    *   Verify your pip installation:
        ```bash
        pip --version
        # or
        pip3 --version
        ```
*   **Git**: Required for cloning repositories if needed (e.g., the Label Studio ML Backend examples). You can download it from [git-scm.com](https://git-scm.com/downloads).
    *   Verify your Git installation:
        ```bash
        git --version
        ```
*   **C++ Compiler (for certain dependencies)**: Some Python packages, especially those used by YOLOv8 or its dependencies, might require a C++ compiler to be installed on your system for building wheels during installation.
    *   **Windows**: Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
    *   **macOS**: Install Xcode Command Line Tools by running `xcode-select --install` in your terminal.
    *   **Linux (Debian/Ubuntu)**: Install `build-essential` by running `sudo apt-get install build-essential` in your terminal.
    *   **Linux (Fedora)**: Install development tools by running `sudo dnf groupinstall "Development Tools"` in your terminal.

It's also highly recommended to work within a **virtual environment** to manage project dependencies and avoid conflicts with other Python projects.

*   **Creating a virtual environment (optional but recommended):**
    ```bash
    # Create a virtual environment named 'ls-yolo-env' (or any name you prefer)
    python3 -m venv ls-yolo-env

    # Activate the virtual environment
    # On macOS and Linux:
    source ls-yolo-env/bin/activate
    # On Windows:
    .\ls-yolo-env\Scripts\activate
    ```
    You'll need to activate the virtual environment in your terminal session before installing any packages.

## 3. Label Studio Installation

With your virtual environment activated (if you created one), you can install Label Studio using pip:

1.  **Install Label Studio:**
    ```bash
    pip install label-studio
    ```
    This command will download and install the latest stable version of Label Studio and its dependencies.

2.  **Verify Installation (Optional):**
    You can check if Label Studio installed correctly by running:
    ```bash
    label-studio --version
    ```

3.  **Launch Label Studio:**
    Once installed, you can start the Label Studio server:
    ```bash
    label-studio start
    ```
    This command will initialize the server, and by default, Label Studio will be accessible at `http://localhost:8080` in your web browser.

    When you first launch Label Studio, you may be prompted to create an account. This account is local to your Label Studio instance.

## 4. YOLOv8 Installation

YOLOv8 is developed by Ultralytics and can be easily installed using pip. It's recommended to install it within the same virtual environment where you installed Label Studio if you are using one.

1.  **Install Ultralytics (YOLOv8):**
    The `ultralytics` package includes YOLOv8 and its dependencies.
    ```bash
    pip install ultralytics
    ```
    This will install the latest version of YOLOv8. If you need a specific version, you can specify it (e.g., `pip install ultralytics==8.0.0`).

2.  **Verify Installation (Optional):**
    You can verify the installation by running a simple YOLO command in your terminal or by importing it in a Python script.
    *   **Terminal Check:**
        ```bash
        yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
        ```
        This command will download the `yolov8n.pt` model (a small, fast version of YOLOv8) if it's not already present and run prediction on a sample image. You should see output in your terminal and an image with detections saved in a `runs/detect/predict` directory.
    *   **Python Script Check:**
        Create a simple Python script (e.g., `check_yolo.py`):
        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLOv8n model
        model = YOLO('yolov8n.pt')

        # Perform inference on an image
        results = model('https://ultralytics.com/images/bus.jpg')

        # Print results (optional)
        for r in results:
            print(r.boxes)

        print("YOLOv8 installation successful!")
        ```
        Run the script:
        ```bash
        python check_yolo.py
        ```
        If the script runs without errors and prints output, YOLOv8 is correctly installed.

With both Label Studio and YOLOv8 installed, the next step is to set up the Label Studio ML Backend to bridge these two tools.

## 5. Label Studio ML Backend Setup

The Label Studio ML Backend is a crucial component that allows you to connect machine learning models (like YOLOv8) to your Label Studio instance for tasks such as automatic annotation. It acts as a bridge, running your model as a web service that Label Studio can communicate with.

### 5.1 Installing the ML Backend

1.  **Install the Label Studio SDK and ML Backend library:**
    If you haven't already (it might have been installed as a dependency of `label-studio`), ensure you have the necessary Label Studio SDK. The `label-studio-ml-sdk` is required to develop your own ML backend.
    ```bash
    pip install label-studio-ml-sdk
    ```

2.  **Choose or Create a Directory for Your ML Backend:**
    You'll need a dedicated directory to store your ML backend script(s) and any associated files (like model weights, if not downloaded automatically).
    For this guide, let's create a directory named `ls_yolo_backend`:
    ```bash
    mkdir ls_yolo_backend
    cd ls_yolo_backend
    ```
    You will create the actual YOLOv8 backend script inside this directory in the next step.

3.  **Initialize an ML Backend (Optional but good practice):**
    Label Studio provides a command to initialize a new ML backend project with a basic structure. This can be helpful as it creates a sample script and a `Dockerfile` (though we are not using Docker for deployment in this guide, the script structure is useful).
    From within your `ls_yolo_backend` directory, you can run:
    ```bash
    label-studio-ml init my_yolo_backend
    ```
    This will create a subdirectory `my_yolo_backend` with a sample `model.py` and other files. You can then adapt this `model.py` or replace it with your custom YOLOv8 script as we will do in the next section. For this guide, we will create our own script from scratch, so you can choose to skip this `init` step if you prefer to create `yolov8_backend.py` directly inside `ls_yolo_backend`.

    If you run `init`, the script we will create in the next step (`yolov8_backend.py`) should be placed inside the `my_yolo_backend` directory (or whatever name you gave it), or directly in `ls_yolo_backend` if you skipped the `init`. For simplicity, this guide will assume the script is named `yolov8_backend.py` and is located in a directory that will be specified when launching the backend (e.g., `ls_yolo_backend/yolov8_backend.py` or `ls_yolo_backend/my_yolo_backend/yolov8_backend.py` if you used `init` and replaced the sample).

### 5.2 Creating the YOLOv8 Backend Script

Below is the Python script that uses YOLOv8 for pre-annotation.

1.  **Save the script:**
    Create a file named `yolov8_backend.py` inside your ML backend directory (e.g., `ls_yolo_backend/yolov8_backend.py` or `ls_yolo_backend/my_yolo_backend/yolov8_backend.py` if you used `label-studio-ml init`).

    ```python
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
            print(f"Attempting to load YOLO model: {YOLO_MODEL_NAME} on device: {GPU_DEVICE}")
            try:
                self.model = YOLO(YOLO_MODEL_NAME)
                self.model.to(GPU_DEVICE)
                print(f"Successfully loaded YOLO model: {YOLO_MODEL_NAME} on device: {GPU_DEVICE}")
            except Exception as e:
                print(f"Error loading YOLO model: {e}")
                self.model = None

            try:
                if hasattr(self.model, 'names') and isinstance(self.model.names, dict):
                    self.labels = [self.model.names[i] for i in range(len(self.model.names))]
                elif hasattr(self.model, 'names') and isinstance(self.model.names, list):
                    self.labels = self.model.names
                else:
                    self.labels = YOLO_MODEL_LABELS
                print(f"Using labels: {self.labels}")
            except Exception as e:
                print(f"Could not get labels from model, using default. Error: {e}")
                self.labels = YOLO_MODEL_LABELS

            if not self.labels:
                print("Warning: No labels found or loaded. Predictions may not have correct class names.")
                self.labels = [f"label_{i}" for i in range(80)]

        def predict(self, tasks, **kwargs):
            if not self.model:
                print("YOLO Model not loaded. Skipping prediction.")
                return []

            predictions = []
            for task in tasks:
                image_url = task['data']['image']
                try:
                    image_path = get_image_local_path(image_url, project_dir=self.project_dir)
                    pil_image = Image.open(image_path)
                    original_width, original_height = pil_image.size
                    results = self.model.predict(image_path, conf=CONFIDENCE_THRESHOLD, device=GPU_DEVICE)

                    task_predictions = []
                    for result in results:
                        for i, box in enumerate(result.boxes.xyxyn):
                            conf = float(result.boxes.conf[i])
                            class_id = int(result.boxes.cls[i])
                            label = self.labels[class_id] if class_id < len(self.labels) else f"class_{class_id}"
                            x_min_norm, y_min_norm, x_max_norm, y_max_norm = box.tolist()

                            prediction_item = {
                                "id": str(uuid4()),
                                "from_name": "label",
                                "to_name": "image",
                                "type": "rectanglelabels",
                                "original_width": original_width,
                                "original_height": original_height,
                                "image_rotation": 0,
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
            print("Fit method called, but not implemented for this YOLOv8 inference backend.")
            pass

    # (The __main__ part of the script can be omitted from the README for brevity if preferred)
    ```

2.  **Understanding the script:**
    *   `YOLO_MODEL_NAME`: Specifies which YOLOv8 model to use (e.g., `yolov8n.pt`). You can change this to other models like `yolov8s.pt`, `yolov8m.pt`, etc., or a path to your custom `.pt` file.
    *   `YOLO_MODEL_LABELS`: A list of class names. This **must** match the classes your YOLO model predicts. The provided list is for standard COCO-trained models. If you use a custom model, update this list accordingly. The script attempts to get labels from `model.names` but falls back to this list.
    *   `CONFIDENCE_THRESHOLD`: Filters detections below this score.
    *   `GPU_DEVICE`: Set to `'cpu'` or a GPU index like `0`.
    *   The `predict` method takes tasks from Label Studio, loads images, runs YOLOv8 inference, and converts the results into Label Studio's required JSON format for bounding boxes.
    *   `from_name` ("label") and `to_name` ("image") in the prediction output must match the names used in your Label Studio labeling configuration.

3.  **Install dependencies for the script:**
    Ensure you have `ultralytics` and `Pillow` (for image processing) in your Python environment where the ML backend will run. If you followed the YOLOv8 installation step, `ultralytics` should already be there. `Pillow` is usually a dependency of `ultralytics` or `label-studio`.
    ```bash
    pip install Pillow
    ```

## 6. Configuring YOLOv8 in Label Studio

Now that you have the YOLOv8 backend script, you need to run it and tell Label Studio how to communicate with it.

### 6.1 Launching the ML Backend

1.  **Navigate to your ML backend directory:**
    Open a new terminal window/tab. Make sure your virtual environment (e.g., `ls-yolo-env`) is activated in this terminal. Navigate to the directory where you saved `yolov8_backend.py` (e.g., `cd ls_yolo_backend`).

2.  **Start the ML backend server:**
    Use the `label-studio-ml start` command. If `yolov8_backend.py` is inside a subdirectory (e.g., `my_yolo_backend` from the `init` command), make sure to point to that directory.
    If `yolov8_backend.py` is directly in `ls_yolo_backend`:
    ```bash
    label-studio-ml start ls_yolo_backend
    ```
    If you used `init` and it's in `ls_yolo_backend/my_yolo_backend/model.py` (and you renamed `model.py` to `yolov8_backend.py` or modified it):
    ```bash
    label-studio-ml start ls_yolo_backend/my_yolo_backend
    ```
    The script will be loaded, and the YOLOv8 model will be downloaded/loaded into memory. You should see output indicating the server is running, usually on `http://localhost:9090`.

    **Important:**
    *   Keep this terminal window open. The ML backend needs to be running whenever you want to use it for pre-annotation in Label Studio.
    *   You can customize the host, port, and other settings. For example, to run on a different port:
        `label-studio-ml start ls_yolo_backend --port 9091`
    *   The first time you run it, the YOLOv8 model (`.pt` file) will be downloaded if it's not already cached by `ultralytics`.
    *   You can set environment variables like `YOLO_MODEL_NAME`, `CONFIDENCE_THRESHOLD`, `GPU_DEVICE` in this terminal before running the `start` command to customize the backend's behavior without modifying the script directly. For example:
        ```bash
        export YOLO_MODEL_NAME='yolov8s.pt' # Linux/macOS
        # set YOLO_MODEL_NAME=yolov8s.pt # Windows
        export CONFIDENCE_THRESHOLD=0.5
        label-studio-ml start ls_yolo_backend
        ```

### 6.2 Connecting Label Studio to the ML Backend

1.  **Open Label Studio:**
    If it's not already running, start Label Studio in another terminal:
    ```bash
    label-studio start
    ```
    And open it in your browser (usually `http://localhost:8080`).

2.  **Create or Open a Project:**
    Create a new project or go to an existing one.

3.  **Go to Project Settings:**
    In your project, click on "Settings" in the top right.

4.  **Navigate to "Machine Learning":**
    In the settings sidebar, choose "Machine Learning".

5.  **Add Model (ML Backend):**
    Click the "+ Add Model" button.

6.  **Configure the ML Backend URL:**
    *   **Title:** Give your backend a descriptive name (e.g., "YOLOv8 Detector").
    *   **URL:** Enter the URL where your ML backend is running. If you started it with default settings, this will be `http://localhost:9090`.
    *   **Description (Optional):** Add any notes.
    *   Toggle "Use for interactive pre-annotations" if you want suggestions as you label.

7.  **Save:**
    Click "Save Model". Label Studio will try to connect to your ML backend. If successful, you'll see it listed with a green checkmark or status indicating it's connected. If it fails, double-check the URL and ensure your ML backend server is running and accessible.

### 6.3 Setting up the Labeling Interface

For the ML backend predictions (bounding boxes) to be displayed correctly, your Label Studio project's labeling interface needs to be configured to handle `rectanglelabels`.

1.  **Go to Labeling Interface:**
    In Project Settings, go to "Labeling Interface".

2.  **Use a Template or Customize:**
    *   You can browse templates and choose "Computer Vision" > "Object Detection with Bounding Boxes".
    *   Or, you can click "Customize" and provide your own XML configuration.

3.  **XML Configuration for Bounding Boxes:**
    Ensure your labeling configuration includes an `<Image>` tag for displaying images and a `<RectangleLabels>` tag for the bounding box labels. The `name` attributes of these tags **must** match the `to_name` and `from_name` values used in your `yolov8_backend.py` script.

    Here's a basic example:
    ```xml
    <View>
      <Image name="image" value="$image" zoom="true" zoomControl="true"/>
      <RectangleLabels name="label" toName="image">
        <!-- Add your labels here. These should ideally match the labels your YOLO model can predict. -->
        <!-- The script currently uses COCO labels by default. -->
        <!-- You can pre-fill some common ones or all of them. -->
        <Label value="person" background="green"/>
        <Label value="car" background="blue"/>
        <Label value="truck" background="orange"/>
        <Label value="traffic light" background="red"/>
        <!-- Add other labels as needed -->
      </RectangleLabels>
    </View>
    ```
    **Key points for the XML:**
    *   `<Image name="image" ...>`: The `name="image"` matches `to_name="image"` in the script.
    *   `<RectangleLabels name="label" toName="image">`: The `name="label"` matches `from_name="label"` in the script.
    *   Inside `<RectangleLabels>`, you define the possible labels using `<Label>` tags. The `value` attribute of each `<Label>` should be one of the class names your YOLO model predicts (and is listed in `YOLO_MODEL_LABELS` or `model.names` in your script).
    *   Providing the list of `<Label>` tags here helps Label Studio display the correct label names and allows users to manually create/edit these labels.

4.  **Save Changes:**
    Click "Save" to update the labeling interface.

## 7. Usage Guide

Once Label Studio, YOLOv8, the ML backend, and the labeling interface are all set up, you can start using the system for automatic pre-annotation.

### 7.1 Creating a Project

If you haven't already, create a new project in Label Studio:
1.  Click "Create Project" on the Label Studio main page.
2.  Give your project a name and an optional description.
3.  Under "Data Import", you can upload some initial data now or do it later.
4.  Under "Labeling Setup", ensure your object detection labeling interface (configured in step 6.3) is selected.
5.  Click "Create Project".

### 7.2 Importing Data

1.  Open your project.
2.  Click the "Import" button to add your images.
3.  You can upload files directly, import from a URL, or connect to cloud storage.
    *   For local files, ensure Label Studio and the ML backend can access them. If you used `get_image_local_path` with `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=True` and `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` properly set up for Label Studio, it should work. Otherwise, using cloud storage or direct uploads might be more straightforward. (Refer to Label Studio documentation for advanced file serving).

### 7.3 Automatic Pre-annotation

With your ML backend connected and running:

1.  **From the Data Manager:**
    *   Go to your project's data manager view (the default view showing your imported tasks/images).
    *   Select the tasks (images) you want to pre-annotate. You can select all or specific ones.
    *   Click the "▼" dropdown on the "Label" button (or it might be directly visible as "Predict" or "Retrieve Predictions" depending on your Label Studio version and setup).
    *   Choose your connected ML backend (e.g., "YOLOv8 Detector") from the list to trigger pre-annotation for the selected tasks.
    *   Label Studio will send the selected tasks to your ML backend, and the backend will return the predictions. This may take some time depending on the number of images and the model's speed.

2.  **Interactive Pre-annotation (if enabled):**
    *   If you enabled "Use for interactive pre-annotations" when adding the ML backend, predictions might automatically load when you open a task in the labeling view.

3.  **Viewing Predictions:**
    *   Once predictions are retrieved, tasks that have annotations will typically show an updated status or display the bounding boxes directly in the data manager if your view is configured for it.
    *   Open a task by clicking "Label" to see the pre-annotations. The YOLOv8-generated bounding boxes and labels should appear on the image.

### 7.4 Reviewing and Correcting Annotations

Automatic pre-annotations are rarely perfect. The next step is to review and correct them:

1.  **Open a Task:**
    Navigate to the labeling view for a task that has been pre-annotated.

2.  **Review Bounding Boxes:**
    *   Check if the bounding boxes correctly cover the objects.
    *   Adjust the size and position of boxes as needed by dragging their edges or corners.
    *   Delete any incorrect or unwanted boxes (e.g., select a box and press Backspace or use the UI delete button).

3.  **Review Labels:**
    *   Verify that the assigned labels are correct.
    *   If a label is wrong, select the bounding box and choose the correct label from the list defined in your labeling interface.

4.  **Add Missing Annotations:**
    *   If the model missed any objects, manually draw new bounding boxes and assign the correct labels.

5.  **Save/Submit:**
    *   Once you are satisfied with the annotations for a task, click "Submit" (or "Update" if it was already submitted) to save your work. This corrected data can then be used to re-train your model if that's part of your workflow.

This workflow significantly speeds up the labeling process by providing a strong starting point, reducing manual effort.

## 8. Troubleshooting

Here are some common issues you might encounter and how to resolve them:

*   **ML Backend Connection Issues:**
    *   **"Cannot connect to ML backend" / "ML backend not responding":**
        *   Ensure your ML backend server is running in a separate terminal (`label-studio-ml start ...`). Check for any error messages in that terminal.
        *   Verify the URL in Label Studio (Project Settings > Machine Learning > Edit Model) matches the host and port your ML backend is listening on (default `http://localhost:9090`).
        *   Check firewall settings: Your firewall might be blocking connections to the ML backend port.
        *   If you changed the port for the ML backend, ensure Label Studio is using the new port.
    *   **ML Backend starts but Label Studio can't fetch predictions:**
        *   Look for errors in the ML backend terminal output when Label Studio tries to get predictions. This often reveals issues within the `predict` method of your `yolov8_backend.py` script.
        *   Make sure the `from_name` and `to_name` in your script's output match the `<RectangleLabels name="...">` and `<Image name="...">` in your Label Studio labeling config.

*   **YOLOv8 Model Issues:**
    *   **Model not downloading / `FileNotFoundError` for `.pt` file:**
        *   Ensure you have an active internet connection when the backend starts for the first time, as it needs to download the model specified by `YOLO_MODEL_NAME`.
        *   If using a custom model path, verify the path is correct and accessible by the Python script.
        *   Ultralytics models are typically saved to `~/Library/Application Support/Ultralytics` on macOS, `~/.config/Ultralytics` on Linux, or `C:\Users\YourUser\AppData\Roaming\Ultralytics` on Windows. Check if the model exists there.
    *   **Incorrect detections / No detections:**
        *   Adjust `CONFIDENCE_THRESHOLD` in `yolov8_backend.py` or via environment variable. A lower threshold means more (potentially incorrect) detections; a higher threshold means fewer (potentially missing some) detections.
        *   Ensure `YOLO_MODEL_LABELS` in `yolov8_backend.py` correctly matches the classes your model was trained on. If the script is fetching labels from `model.names`, verify those are correct.
        *   If using a custom model, make sure it's suitable for the kind of images you're labeling.
        *   Check the ML backend terminal for any runtime errors from the YOLO model.
    *   **CUDA / GPU issues (`RuntimeError: CUDA out of memory`, etc.):**
        *   If you don't have a compatible NVIDIA GPU or CUDA installed correctly, set `GPU_DEVICE = 'cpu'` in `yolov8_backend.py` or via environment variable `export GPU_DEVICE='cpu'`.
        *   If you have a GPU but it's running out of memory, try a smaller YOLO model (e.g., `yolov8n.pt` instead of `yolov8x.pt`), reduce batch size if applicable (not directly configurable in this basic script, but relevant for custom modifications), or close other GPU-intensive applications.
        *   Ensure your PyTorch installation is compatible with your CUDA version. The `ultralytics` package tries to handle this, but issues can arise.

*   **Label Studio Display Issues:**
    *   **Bounding boxes not showing up:**
        *   Verify the `from_name` in the ML backend's prediction output matches the `<RectangleLabels name="your_label_name">` in the labeling config.
        *   Verify the `to_name` matches the `<Image name="your_image_name">`.
        *   Check the structure of the prediction JSON in the ML backend terminal. It must conform to what Label Studio expects for `rectanglelabels`.
    *   **Labels are incorrect or all generic (e.g., "class_0", "class_1"):**
        *   This means `YOLO_MODEL_LABELS` list in `yolov8_backend.py` is not being used correctly or the model's internal class names are not what you expect. Double-check this list and ensure it aligns with your model's output classes.
        *   If your model is supposed to have embedded class names (`model.names`), ensure the script is accessing them correctly.

*   **Python Environment & Dependencies:**
    *   **`ModuleNotFoundError` (e.g., `No module named 'ultralytics'` or `No module named 'label_studio_ml'`):**
        *   Make sure you have activated the correct virtual environment (if using one) in the terminal where you run `label-studio-ml start`.
        *   Install the missing package using `pip install <package_name>`.
    *   **Version conflicts:**
        *   Using a virtual environment helps minimize these. If you encounter persistent conflicts, you might need to create a fresh environment and install packages one by one, checking compatibility.

*   **Image Access Issues by ML Backend:**
    *   **`FileNotFoundError` for image path in `get_image_local_path`:**
        *   This is common if Label Studio is serving images from a directory that the ML backend script can't directly access using the same path.
        *   For `get_image_local_path` to work reliably for local files, Label Studio needs to be started with local file serving enabled, and the `project_dir` argument in `get_image_local_path` might need to be configured if your ML backend isn't in the Label Studio project directory.
        *   Simpler options: Use cloud storage URLs for images, or ensure images are in a location accessible by both Label Studio and the ML backend process with the same path.
        *   When testing the backend script directly (`python yolov8_backend.py`), image URLs will be downloaded by `get_image_local_path`. When run as an ML backend, it depends on how Label Studio provides the URL.

*   **General Tips:**
    *   **Check Terminal Outputs:** Always keep an eye on the terminal running Label Studio and the terminal running the ML backend. Error messages are your best friend for debugging.
    *   **Start Simple:** If things go wrong, revert to the simplest configuration (e.g., `yolov8n.pt`, COCO labels, CPU processing) to see if that works, then add your customizations back one by one.
    *   **Consult Documentation:** Refer to the [Label Studio documentation](https://labelstud.io/guide/) and [Ultralytics YOLOv8 documentation](https://docs.ultralytics.com/) for more detailed information.

## 9. Conclusion
(Summary and final thoughts)
