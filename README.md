# Image Detection & Segmentation Labeler

A suite of lightweight, standalone GUI tools built in Python. This repository contains two dedicated tools: an **Image Labeler** for static bounding box and polygon mask segmentation, and a **Video Labeler** for frame extraction and automated object tracking.

## Features

### Image Labeler (`labeler.py`)
* **Dual Annotation Modes:** Support for both Bounding Box (BBox) and Polygon Mask segmentation, even on the same image.
* **Universal Import:** Automatically loads existing annotations from corresponding `.json` files or YOLO format `.txt` files (from the same folder or a `../labels/` folder).
* **Instant Probe:** Right-click anywhere on the image to get instant X/Y pixel coordinates.

### Video Labeler (`video_labeler.py`)
* **Direct Video Annotation:** Load video files directly (`.mp4`, `.avi`, `.mkv`, or `.mov`), play, pause, and step frame-by-frame to annotate before extracting.
* **Advanced Auto-Tracking:** Uses OpenCV's CSRT tracker combined with strict Structural Similarity (SSIM) and edge-validation to rigorously follow objects and automatically drop them to prevent background drift.
* **Smart Jump Recovery:** Automatically searches the immediate surroundings using template matching to re-lock onto fast-moving objects that jump between consecutive frames.
* **End-of-Workflow Extraction:** Extracts frames and labels at a user-defined FPS into a smart `[video_name]_extracted` directory only after you finish tracking. 
* **Seamless State Management:** Automatically detects if an extracted folder exists for a video and seamlessly switches to "Extracted Mode" for easy label editing or extraction removal.

### Shared Features
* **Dynamic Categories:** Load custom class lists directly from the GUI (e.g., `class_name = id`).

---

## Installation & Setup

### Option 1: Running from Source
Ensure you have Python installed on your system. First, install the required dependencies:

~~~bash
pip install -r requirements.txt
~~~

Once the dependencies are installed, you can launch either tool directly from your terminal:

~~~bash
python labeler.py
# OR
python video_labeler.py
~~~

### Option 2: Creating Standalone Executables
If you want to create double-clickable applications that do not require Python to be installed on the host machine, use PyInstaller:

~~~bash
pip install pyinstaller Pillow opencv-python numpy
~~~

Then, generate the executables:

~~~bash
pyinstaller --noconsole --onefile labeler.py
pyinstaller --noconsole --onefile video_labeler.py
~~~

Look inside the newly generated `dist/` folder to find your standalone executables.

---

## How to Use the Image Labeler

1. **Load Classes (Optional):** Click **"Load Classes"** to import a `.txt` file. If a folder contains `categories.txt` or `classes.txt`, it loads automatically.
2. **Load Folder:** Click **"Load Folder"** and select the directory containing your `.png`, `.jpg`, or `.jpeg` images.
3. **Select Mode & Category:** Choose the desired class category and toggle between **BBox** or **Mask** mode.
4. **Draw Annotations:**
    * **BBox:** Click and drag to draw a bounding box.
    * **Mask:** Click sequentially to place polygon vertices. Click on the first point of the mask to close and finalize the polygon.
5. **Manage Annotations:** Use the **Clear Last (L)** button to undo, or use the sidebar to change classes/delete annotations.
6. **Save & Navigate:** Click **Save JSON (S)** to save. Use **Next (D)** and **Prev (A)** to navigate.

---

## How to Use the Video Labeler

1. **Load Video Folder:** Click **"Load Video Folder"** and select a directory containing your source videos. The tool will load the video (Video Mode) or default to the extracted frames if they already exist (Extracted Mode).
2. **Navigate & Play:** Use the **▶ Play** button, the timeline slider, or the **⏮** / **⏭** buttons (or **A** / **D** keys) to step through the video frame-by-frame.
3. **Draw Initial Annotation:** Toggle between **BBox** or **Mask** mode. Click and drag to draw a bounding box around the object, or click sequentially to draw a polygon mask (click the first point to close it).
4. **Auto-Track Video:** 
    * Click **"Auto-Track Video"**. 
    * The algorithm will automatically start from the first frame, scrub forward to find your labels, and track the objects to the end of the video using SSIM validation and Jump Recovery. *(Note: Masks cannot be auto-tracked, only bounding boxes).*
5. **Extract & Save:** 
    * Click **"Extract Frames & Save (S)"**.
    * Enter your desired extraction FPS (e.g., 5 for 5 frames per second).
    * The tool will generate the images and `.json` label files into a new extracted folder and automatically switch you to Extracted Mode.
6. **Modify or Remove:** In Extracted Mode, you can edit individual labels and click **"Save JSON (S)"** to update the files. If you need to restart the tracking process from scratch, click **"Remove Extracted Folder"**.

---

## Outputs

Both tools generate a `.json` file in the same directory as the image, sharing the exact same base file name.

### JSON Format Structure

~~~json
{
    "type": "mixed_annotations",
    "date": "20260716_143022",
    "image": {
        "file_name": "example_image.png",
        "width": 1920,
        "height": 1080
    },
    "annotations": [
        {
            "id": 1,
            "category_id": 0,
            "xmin": 150,
            "ymin": 200,
            "width": 300,
            "height": 450,
            "type": "bbox"
        },
        {
            "id": 2,
            "category_id": 1,
            "segmentation": [
                [
                    500.0, 600.0,
                    550.0, 620.0,
                    520.0, 680.0
                ]
            ],
            "type": "segmentation"
        }
    ]
}
~~~

* **BBox coordinates** are saved as absolute pixel values (`xmin`, `ymin`, `width`, `height`).
* **Segmentation coordinates** are saved as a flat list of absolute pixel values `[x1, y1, x2, y2, ...]`.
