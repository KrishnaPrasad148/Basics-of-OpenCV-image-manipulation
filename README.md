# Basics-of-OpenCV-image-manipulation


## AIM:
Write a Python program using OpenCV that performs the following tasks:

1) Read and Display an Image.  
2) Adjust the brightness of an image.  
3) Modify the image contrast.  
4) Generate a third image using bitwise operations.

## Software Required:
- Anaconda - Python 3.7
- Jupyter Notebook (for interactive development and execution)

## Algorithm:
### Step 1:
Load an image from your local directory and display it.

### Step 2:
Create a matrix of ones (with data type float64) to adjust brightness.

### Step 3:
Create brighter and darker images by adding and subtracting the matrix from the original image.  
Display the original, brighter, and darker images.

### Step 4:
Modify the image contrast by creating two higher contrast images using scaling factors of 1.1 and 1.2 (without overflow fix).  
Display the original, lower contrast, and higher contrast images.

### Step 5:
Split the image (boy.jpg) into B, G, R components and display the channels

## Program Developed By:
- **Name:** Krishna Prasad S
- **Register Number:** 212223230108

  ### Ex. No. 01

#### 1. Read the image ('Eagle_in_Flight.jpg') using OpenCV imread() as a grayscale image.
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Eagle_in_Flight.jpg',0)
```

#### 2. Print the image width, height & Channel.
```python
img.shape
```

#### 3. Display the image using matplotlib imshow().
```python
plt.imshow(img)
plt.title("GRAY IMAGE")
plt.axis("on")
plt.show()
```

#### 4. Save the image as a PNG file using OpenCV imwrite().
```python
cv2.imwrite('Grey_Eagle.jpg', img)
```

#### 5. Read the saved image above as a color image using cv2.cvtColor().
```python
clr_img = cv2.imread('Grey_Eagle.jpg',cv2.IMREAD_COLOR)
clr_img = cv2.cvtColor(clr_img, cv2.COLOR_BGR2RGB)
```

#### 6. Display the Colour image using matplotlib imshow() & Print the image width, height & channel.
```python
plt.imshow(clr_img)
plt.title("COLOR IMAGE")
plt.axis("on")
plt.show()
clr_img.shape
```

#### 7. Crop the image to extract any specific (Eagle alone) object from the image.
```python
cropped_img = clr_img[20:415, 200:550]
plt.imshow(cropped_img)
plt.title("CROPPED IMAGE")
plt.axis("on")
plt.show()
```

#### 8. Resize the image up by a factor of 2x.
```python
resized_img = cv2.resize(cropped_img, None, fx = 50, fy = 50, interpolation=cv2.INTER_LINEAR)
resized_img.shape
```

#### 9. Flip the cropped/resized image horizontally.
```python
flip_img = cv2.flip(resized_img, 1)

plt.imshow(flip_img)
plt.title("FLIPPED IMAGE")
plt.axis("off")
plt.show()
```

#### 10. Read in the image ('Apollo-11-launch.jpg').
```python
img2 = cv2.imread('Apollo-11-launch.jpg')

plt.imshow(img2)
plt.title("IMAGE")
plt.axis("on")
plt.show()
```

#### 11. Add the following text to the dark area at the bottom of the image (centered on the image):
```python
text = 'Apollo 11 Saturn V Launch, July 16, 1969'
font_face = cv2.FONT_HERSHEY_PLAIN
font_color = (255, 255, 255)
img3 = img2.copy()
img3 = cv2.putText(img3, text, (90, 700), font_face, 3.2, font_color, 4, cv2.LINE_AA)
```

#### 12. Draw a magenta rectangle that encompasses the launch tower and the rocket.
```python
rect_color = (255, 0, 255)
mag_img = img3.copy()
mag_img = cv2.rectangle(mag_img, (500, 50), (700, 650), rect_color, 10, cv2.LINE_8)
```

#### 13. Display the final annotated image.
```python
plt.imshow(mag_img)
plt.title("IMAGE")
plt.axis("off")
plt.show()
```

#### 14. Read the image ('Boy.jpg').
```python
image = cv2.imread('boy.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

#### 15. Adjust the brightness of the image.
```python
matrix_ones = np.ones(image.shape, dtype = 'uint8') * 30
matrix_ones
```

#### 16. Create brighter and darker images.
```python
img_brighter = cv2.add(image, matrix_ones)
img_darker = cv2.subtract(image, matrix_ones)
```

#### 17. Display the images (Original Image, Darker Image, Brighter Image).
```python
plt.figure(figsize = (18, 5))
plt.subplot(131); plt.imshow(image); plt.title("ORIGINAL")
plt.subplot(132); plt.imshow(img_darker); plt.title("DARKER")
plt.subplot(133); plt.imshow(img_brighter); plt.title("BRGHTER");
```

#### 18. Modify the image contrast.
```python
matrix1 = np.ones(image.shape) * 1.1
matrix2 = np.ones(image.shape) * 1.2
img_higher1 = np.uint8(cv2.multiply(np.float64(image), matrix1))
img_higher2 = np.uint8(cv2.multiply(np.float64(image), matrix2))
```

#### 19. Display the images (Original, Lower Contrast, Higher Contrast).
```python
plt.figure(figsize = (18, 5))
plt.subplot(131); plt.imshow(image); plt.title("ORIGINAL")
plt.subplot(132); plt.imshow(img_higher1); plt.title("LOWER CONTRAST")
plt.subplot(133); plt.imshow(img_higher2); plt.title("HIGHER CONTRAST");
```

#### 20. Split the image (boy.jpg) into the B,G,R components & Display the channels.
```python
b, g, r = cv2.split(image)

plt.figure(figsize = (18, 5))
plt.subplot(141); plt.imshow(r); plt.title("RED CHANNEL")
plt.subplot(142); plt.imshow(g); plt.title("GREEN CHANNEL")
plt.subplot(143); plt.imshow(b); plt.title("BLUE CHANNEL");
```

#### 21. Merged the R, G, B , displays along with the original image
```python
merged_img = cv2.merge((r, g, b))

plt.figure(figsize = (22, 5))
plt.subplot(131); plt.imshow(image); plt.title("ORIGINAL")
plt.subplot(132); plt.imshow(merged_img); plt.title("MERGED");
```

#### 22. Split the image into the H, S, V components & Display the channels.
```python
img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(img_hsv)

plt.figure(figsize = (18,5))
plt.subplot(141); plt.imshow(h); plt.title("H CHANNEL")
plt.subplot(142); plt.imshow(s); plt.title("S CHANNEL")
plt.subplot(143); plt.imshow(v); plt.title("V CHANNEL");
```
#### 23. Merged the H, S, V, displays along with original image.
```python
merged_hsv = cv2.merge((h, s, v))

plt.figure(figsize = (22,5))
plt.subplot(131); plt.imshow(image); plt.title("ORIGINAL")
plt.subplot(132); plt.imshow(merged_hsv); plt.title("MERGED");
```

## Output:
- **i)** Read and Display an Image.  
- **ii)** Adjust Image Brightness.  
- **iii)** Modify Image Contrast.  
- **iv)** Generate Third Image Using Bitwise Operations.

## Result:
Thus, the images were read, displayed, brightness and contrast adjustments were made, and bitwise operations were performed successfully using the Python program.


