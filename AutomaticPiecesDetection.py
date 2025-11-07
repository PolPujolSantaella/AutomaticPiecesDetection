import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt

PATH_IMAGES = "Assignment 1/Images"

def load_images(img_path):
    files = [f for f in os.listdir(img_path) if f.lower().endswith('.png')]
    files.sort()

    images = []
    for filename in files:
        full_path = os.path.join(img_path, filename)
        img = cv2.imread(full_path)
        if img is None:
            print(f"Image unable to be loaded: {full_path}")
            continue
        images.append(img)

    print(f"Loaded {len(images)} images from {img_path}")

    return images

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    img_preprocessed = blurred.copy()

    return img_preprocessed


def detect_circles(img_preprocessed):
    circles = cv2.HoughCircles(
        img_preprocessed,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=50,
        param2=25,
        minRadius=90,
        maxRadius=150
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(f"Detected {circles.shape[1]} circles.")
    else:
        print("No circles detected.")

    return circles


def get_complete_pieces(img_shape, circles):
    if circles is None:
        return 0, None
        
    H, W, _ = img_shape
    complete_circles = []

    for x, y, r in circles[0, :]:
        is_complete_x = (x - r) > 0 and (x + r) < W
        is_complete_y = (y - r) > 0 and (y + r) < H
        
        if is_complete_x and is_complete_y:
            complete_circles.append((x, y, r))
            
    if complete_circles:
        complete_circles_np = np.array(complete_circles, dtype=np.uint16).reshape(1, -1, 3)
        return len(complete_circles), complete_circles_np
    else:
        return 0, None

def draw_circles(img, complete_circles):
    output = img.copy()

    if complete_circles is not None:
        for (x, y, r) in complete_circles[0, :]:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)  # contorno
            cv2.circle(output, (x, y), 2, (0, 0, 255), 3)  # centro
    return output


def main():
    images = load_images(PATH_IMAGES)
    if not images: 
        print("No iamges loaded.")
        return
    
    drawn_images_list = []
    final_counts = []
    preprocessed_images_list = []

    for img in images:
        preprocessed_img = preprocess_image(img)
        detected_circles = detect_circles(preprocessed_img)

        count, complete_circles = get_complete_pieces(img.shape, detected_circles)

        drawn_img = draw_circles(img, complete_circles)

        preprocessed_images_list.append(preprocessed_img)
        drawn_images_list.append(drawn_img)
        final_counts.append(count)

    n = len(images)

    for i, c in enumerate(final_counts, 1):
        print(f"✅ Image {i}: {c} complete pieces detected.")

    # Display images
    fig, axes = plt.subplots(3, n, figsize=(3*n, 8))

    if n == 1:
        axes = np.array(axes).reshape(3, 1)

    for i in range(n):
        axes[0, i].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis('off')

        axes[1, i].imshow(preprocessed_images_list[i], cmap='gray')
        axes[1, i].set_title(f"Preprocessed {i+1}")
        axes[1, i].axis('off')

        axes[2, i].imshow(cv2.cvtColor(drawn_images_list[i], cv2.COLOR_BGR2RGB))
        axes[2, i].set_title(f"Detected (Count: {final_counts[i]})")
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 6))
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.imshow(cv2.cvtColor(drawn_images_list[i], cv2.COLOR_BGR2RGB))
        ax.set_title(f"Image {i+1}\nDetected: {final_counts[i]} pieces",
                     fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.suptitle("Final Results: Detected Pieces per Image", fontsize=18, fontweight='bold', color='darkblue')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show() 

if __name__ == "__main__":
    main()