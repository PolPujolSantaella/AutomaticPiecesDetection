# AutomaticPiecesDetection

## Overview

# 🤖 Automatic Pieces Detection and Quantification in Thermographic Images

## 📋 Overview

This project implements a computer vision pipeline to **automatically detect and quantify complete circular pieces** within thermographic images. The system is designed to process images from a thermal sealing line to accurately count components that are fully visible.

The core pipeline involves:

1.  **Preprocessing:** Converting the image to grayscale and applying Gaussian blur.

2.  **Feature Detection:** Using the **Hough Gradient Method** to detect potential circular pieces and the **Hough Line Transform** (via a custom Sobel-like kernel) to detect the surrounding thermal sealing lines.

3.  **Geometric Filtering:** Applying a custom geometric filter that uses the perpendicular distance between the center of each detected circle and the surrounding lines to determine if the piece is truly "complete" (i.e., not cut off by the frame or the sealing mechanism).

4.  **Visualization:** Generating a detailed multi-panel output showing each stage of the detection process and the final count.

## 📁 Project Structure 
```bash
├──Assignment 1/
    ├──Images/      #Input images
    ├──Task1.pdf    #Assignment description
├──output_images/   #Results
├──.gitignore
├──ArtificialVision_Assignment1_PolPujolSantaella.pdf # Project Documentation
├──AutomaticPiecesDetection.py # Code
├──README.md
├──requirements.txt # List of Python dependencies
```

## 🚀 Getting Started

These instructions will get a copy of the project running on your local machine.

### 1. Environment Setup

It is highly recommended to use a virtual environment to manage dependencies.

1.  **Create a virtual environment (optional but recommended):**
    ```bash
    python3 -m venv venv
    ```

2.  **Activate the environment:**
    * **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
    * **Windows (Command Prompt):**
        ```bash
        venv\Scripts\activate
        ```

3.  **Install dependencies:**
    The project relies on standard scientific and computer vision libraries. Install them using the provided `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
    *(Dependencies include `opencv-python`, `matplotlib`, and `numpy`.)*

### 2. Execution

Run the main Python script. The script will automatically load the images, process them, calculate the final count, and save the visualization results.

```bash
python3 AutomaticPiecesDetection.py
```
### 3. Outputs
Upon successful execution, the script will:

Print the detected count for each image to the console.

Create an output_images folder in the root directory.

Save a multi-panel visualization PNG for each input image in the output_images folder, illustrating the full detection pipeline and the final result.