import cv2 
import os
import numpy as np
import matplotlib.pyplot as plt

# CONFIG
PATH_IMAGES = "Assignment 1/Images"
TARGET_WIDTH = 800
HOUGH_PARAMS = {
    'dp': 1.0,
    'minDist': 200,
    'param1': 50,
    'param2': 45,
    'minRadius': 100,
    'maxRadius': 200
}

KERNEL = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])

def resize_image_aspect(image, target_width):
    """
        Function to resize heigth and maintain aspect ratio.
        Return: The image resized
    """
    h, w = image.shape[:2]
    new_h = int(target_width * h / w)
    return cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_AREA)

def load_images(path):
    """
        Function to read and resize images to append them to a list
        Return: A list of images loaded.
    """
    files = sorted([f for f in os.listdir(path) if f.lower().endswith(('.png'))])
    images = []

    for name in files:
        img = cv2.imread(os.path.join(path, name))
        if img is None:
            print(f"Images unable to be loaded: {name}")
            continue
        images.append(resize_image_aspect(img, TARGET_WIDTH))

    if images:
        h, w = images[0].shape[:2]
        print(f"Loaded {len(images)} images. Resized to {w}x{h}")
    else:
        print("No images loaded.")

    return images

def preprocess_image(img):
    """
        Function to preprocess the image. 
            - Convert to grayscale
            - Apply Gaussian filter (5x5)

        Return: Image preprocessed
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def detect_lines(img, kernel):
    """
        Function to detect horizontal lines using Sobel Y
        Return: List of lines detected
    """

    conv = cv2.filter2D(img, -1, kernel)
    h, w = conv.shape
    lines = [
        np.array([0, h, w, h]), # Bottom border
        np.array([0, 0, 0, h]) # Left border
    ]

    detected_lines = cv2.HoughLinesP(conv, 1, np.pi / 180, 50, None, 400, 0)
    
    if detected_lines is not None:
        lines.extend([l[0] for l in detected_lines])

    return lines

def detect_circles(img, params):
    """
        Function to detect circles using HoughCircles
        Return: Arrays of N circles
    """
    return cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        dp=params['dp'],
        minDist=params['minDist'],
        param1=params['param1'],
        param2=params['param2'],
        minRadius=params['minRadius'],
        maxRadius=params['maxRadius']
    )

def line_point_distance(x, y, x1, y1, x2, y2):
    """
        Function to calculate min distance between a point and a line.
        Return: The min distance
    """
    denom = np.hypot(y2 - y1, x2 - x1)
    if denom == 0:
        return float('inf')
    num = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    return num / denom 

def get_complete_pieces(circles, lines):
    """
        Function to count how many complete circles we have. 
        Using the detected lines, calculate the distance between the center of the circle and the lines.
        If the distance is lower than (radius - 10px) the line is inside the circle

        Return: Count of complete circles, array of complete circles
    """
    if circles is None:
        return 0, np.empty((0, 3), dtype=np.uint16)
    
    circles = np.uint16(np.around(circles))
    complete = []

    for x, y, r in circles[0]:
        is_complete = True
        for x1, y1, x2, y2 in lines:
            if line_point_distance(x, y, x1, y1, x2, y2) - r < -10:
                is_complete = False
                break
        if is_complete:
            complete.append([x, y, r])
    return len(complete), np.array(complete)

def draw_circles(img, circles, lines, count):
    """
        Function to draw the complete circles.
        Return: The image with complete circles drawed.
    """
    output = img.copy()

    for x1, y1, x2, y2 in lines:
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 5, cv2.LINE_AA)

    for x, y, r in circles:
        cv2.circle(output, (x, y), r, (0, 0, 0), 5)
        cv2.circle(output, (x, y), 5, (0, 0, 255), -1)

    cv2.putText(output, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
    return output
    

def main():

    # STEP 1: Load Images
    images = load_images(PATH_IMAGES)
    if not images:
        return
    
    originals = []
    preprocessed_images = []
    line_images = []
    circle_images = []
    drawn_list = []
    final_counts = []

    # For each image
    for img in images:
        originals.append(img.copy())
        
        # STEP 2: Preprocessing
        preprocessed = preprocess_image(img)
        preprocessed_images.append(preprocessed)

        # STEP 3: Detect Horizontal Lines using HoughLinesP
        lines = detect_lines(preprocessed, KERNEL)
        line_vis = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
        for x1, y1, x2, y2 in lines:
            cv2.line(line_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        line_images.append(line_vis)

        # STEP 4: Detect Circles
        circles = detect_circles(preprocessed, HOUGH_PARAMS)

        # STEP 5: Count complete circles
        count, complete = get_complete_pieces(circles, lines)
        final_counts.append(count)

        circ_vis = img.copy()
        if len(complete) > 0:
            for x, y, r in complete:
                cv2.circle(circ_vis, (x, y), r, (255, 0, 0), 4)
        circle_images.append(circ_vis)

        # STEP 6: Draw Complete circles
        drawn_list.append(draw_circles(img, complete, lines, count))

    # STEP 7: Visualization
    for i, c in enumerate(final_counts, 1):
        print(f"Image {i}: {c} complete pieces detected")

    for i in range(len(images)):

        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(f"Image {i+1}", fontsize=18)

        # Original Image
        ax1 = plt.subplot2grid((2,4), (0,0))
        ax1.imshow(cv2.cvtColor(originals[i], cv2.COLOR_BGR2RGB))
        ax1.set_title("Original")
        ax1.axis("off")

        # Preprocessed Image
        ax2 = plt.subplot2grid((2,4), (0,1))
        ax2.imshow(preprocessed_images[i], cmap="gray")
        ax2.set_title("Preprocessed")
        ax2.axis("off")

        # Lines detected
        ax3 = plt.subplot2grid((2,4), (0,2))
        ax3.imshow(cv2.cvtColor(line_images[i], cv2.COLOR_BGR2RGB))
        ax3.set_title("Detected Lines")
        ax3.axis("off")

        # Circles detected
        ax4 = plt.subplot2grid((2,4), (0,3))
        ax4.imshow(cv2.cvtColor(circle_images[i], cv2.COLOR_BGR2RGB))
        ax4.set_title("Complete Circles Only")
        ax4.axis("off")

        # Final Result
        ax5 = plt.subplot2grid((2,4), (1,0), colspan=4)
        ax5.imshow(cv2.cvtColor(drawn_list[i], cv2.COLOR_BGR2RGB))
        ax5.set_title(f"Final Detection (Count = {final_counts[i]})")
        ax5.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

if __name__ == "__main__":
    main()