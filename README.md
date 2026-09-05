# EXP - 7-  Record-HOUGH-TRANSFORM

# **Name:** Vignesh S 

# **Register No:** 212223230240

---

## Aim

To implement a basic edge detection pipeline using OpenCV by completing missing code segments at specified locations.

---

## Learning Objective

* Understand each stage of image processing
* Learn how to build a complete computer vision pipeline
* Practice writing code in guided sections


---

## Software Used

* Anaconda – Python 3.7
* Jupyter Notebook / VS Code
* OpenCV (cv2)
* NumPy
* Matplotlib

---

## Algorithm & Explanation

---

### Step 1: Import Libraries

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

---

### Step 2: Read the Image

```python


# Read the image using OpenCV
image = cv2.imread('highway lane.jpg')

if image is None:
    raise FileNotFoundError("Image not found. Keep building.jpg in the same folder as this notebook.")

original_image = image.copy()

```

---

### Step 3: Convert to Grayscale

```python
# Convert to grayscale.


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



```
```
 Input image and grayscale image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert image to RGB for displaying
plt.title("Input Image")
plt.axis('off')
---
```
plt.imshow(gray_image, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
```

### Step 4: Display Images

```python
plt.figure(figsize=(10,5))

```
```
plt.imshow(gray_image, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off'
```
```

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")
```
```
plt.subplot(1, 2, 2)
plt.imshow(gray_image, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.show()
```

---

### Step 5: Thresholding

```
threshold = cv2.threshold(
    gray_image,
    127,
    255,
    cv2.THRESH_BINARY
)[1]
```
```

plt.imshow(threshold, cmap="gray")
plt.title("Thresholded Image")
plt.axis("off")
plt.show()
```

---

### Step 6: Region of Interest (ROI)

```
# ROI masking already provided
height, width = threshold.shape
```
```
# Define a broad polygonal region of interest
roi_vertices = np.array([[
    (0, height),
    (0, int(height * 0.45)),
    (width, int(height * 0.45)),
    (width, height)
]], dtype=np.int32)
```
```

roi_mask = np.zeros_like(threshold)
cv2.fillPoly(roi_mask, roi_vertices, 255)
roi_image = cv2.bitwise_and(threshold, roi_mask)
```

---

### Step 7: Edge Detection (Canny)

```
# Perform Edge Detection
```
```
# Canny Edge Detector output
plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detector")
plt.axis('off')
```
```

edges = cv2.Canny(
    gray_image,
    50,
    150
)
```
```
plt.imshow(edges, cmap="gray")
plt.title("Canny Edge Detection")
plt.axis("off")
plt.show()
```

---

### Step 8: Gaussian Blur

```
# Apply Gaussian Blur


```
```
blurred = cv2.GaussianBlur(edges, (5, 5), 0)
```
```
plt.figure(figsize=(8, 10))
plt.imshow(blurred, cmap='gray')
plt.title("Smoothed Image")
plt.axis('off')
plt.show()
```
```

blurred = cv2.GaussianBlur(
    edges,
    (5, 5),
    0
)
```
```
plt.imshow(blurred, cmap="gray")
plt.title("Smoothed Image")
plt.axis("off")
plt.show()
```

---

### Step 9: Hough Transform

```
# Detect lines using Hough Transform

```
```
lines = cv2.HoughLinesP(
    edges,
    1,
    np.pi / 180,
    100,
    minLineLength=50,
    maxLineGap=10
)
```
```
for line in lines:
    x1, y1, x2, y2 = line[0]  # Access the 0th element to unpack the 4 values
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    ```
    ```
    # Display the result of Hough Transform (Image with lines)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Image with lines drawn
plt.title("Result of Hough Transform")
plt.axis('off')
```
```


lines = cv2.HoughLinesP(
    blurred,
    1,
    np.pi / 180,
    threshold=100,
    minLineLength=50,
    maxLineGap=10
)
```
```



### Step 10: Edge Detection Logic

```python
# Already implemented
# Step 5: Detect lines using HoughLinesP
lines = cv2.HoughLinesP(
    edges,
    1,
    np.pi / 180,
    100,
    minLineLength=50,
    maxLineGap=10
)
```
```
# Step 6: Using a for loop, draw the lines on the original image using the detected coordinates
# The lines variable contains the endpoints of the detected lines
if lines is not None:
    print(lines.shape)
    print(lines[:5])
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
# Display the result of Hough Transform (Image with lines)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Image with lines drawn
plt.title("Result of Hough Transform")
plt.axis('off')


```

---

## Expected Output
### Original image

<img width="516" height="381" alt="image" src="https://github.com/user-attachments/assets/b123f399-e56a-40ae-b23d-c29b405ca0a3" />

  
### Grayscale image

<img width="516" height="381" alt="image" src="https://github.com/user-attachments/assets/25955d94-fce3-4487-8b6d-55c8c7b24c10" />


### Thresholded image

<img width="640" height="466" alt="image" src="https://github.com/user-attachments/assets/919bd0a6-f171-4afc-a09d-949870a6ec95" />


### ROI masked image
<img width="640" height="466" alt="image" src="https://github.com/user-attachments/assets/9697c178-6db3-44ad-b94a-2f26aeafd0ac" />


### Edge detected image

<img width="516" height="381" alt="image" src="https://github.com/user-attachments/assets/173f8529-ea81-47e7-9942-814b2c45534a" />

### Smoothed image

<img width="640" height="466" alt="image" src="https://github.com/user-attachments/assets/c143d6bf-93b5-4828-b7c3-63a5c424357e" />


### Detected lines

```
(373, 4)
[[146 732 146  57]
 [148 769 148 100]
 [152 774 152  58]
 [158 771 158  59]
 [ 94 575  98 330]]
```

### Final edge detection output

<img width="516" height="381" alt="image" src="https://github.com/user-attachments/assets/cb5ed465-c1c6-481c-abd2-169ff5856226" />


## Result

Thus, the edge detection pipeline is successfully implemented by completing the missing code sections. The system detects and highlights edges and lines effectively.


