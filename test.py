import cv2

image_path = "/Users/naveenkumar/Downloads/UNET/test_images/sample.jpg"

img = cv2.imread(image_path)
if img is None:
    print("Error: OpenCV failed to load the image.")
else:
    print("Image loaded successfully!")
