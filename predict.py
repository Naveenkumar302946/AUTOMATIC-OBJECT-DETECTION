import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# Load trained model
MODEL_PATH = "models/unet_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Image path (replace with an actual test image)
TEST_IMAGE_PATH = "/Users/naveenkumar/Downloads/UNET/test_images/sample.jpg"

# Image size
IMG_HEIGHT = 256
IMG_WIDTH = 256

# Load and preprocess test image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError(f"Error: Could not load image at {image_path}. Check file format and path.")

    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0  # Normalize
    return img

# Predict segmentation mask
def predict_mask(image_path):
    img = preprocess_image(image_path)
    img_input = np.expand_dims(img, axis=0)  # Add batch dimension
    
    predicted_mask = model.predict(img_input)[0]  # Get the first output
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)  # Threshold to binary mask
    
    return img, predicted_mask

# Overlay mask on original image
import cv2
import numpy as np

def overlay_mask(original, mask):
    """Overlay a black-and-white segmentation mask on the original image."""
    
    if mask is None or mask.size == 0:
        raise ValueError("Error: The segmentation mask is empty!")

    mask = mask.squeeze()  # Remove extra dimensions
    
    if len(mask.shape) != 2:
        raise ValueError(f"Error: Expected a 2D mask, got shape {mask.shape}")

    print(f"Original Image Shape: {original.shape}, Mask Shape Before Resize: {mask.shape}")

    # Ensure mask has the same size as the original image
    mask_resized = cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Convert mask to 3-channel grayscale (for OpenCV blending)
    mask_rgb = cv2.cvtColor((mask_resized * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Blend original image with the mask
    overlay = cv2.addWeighted(original, 0.7, mask_rgb, 0.3, 0)

    return overlay


# Run prediction
original, mask = predict_mask(TEST_IMAGE_PATH)

# Convert original from BGR to RGB (for Matplotlib display)
original_rgb = cv2.cvtColor((original * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)

# Generate overlay
overlayed_image = overlay_mask(original_rgb, mask)

# Display results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(original_rgb)
plt.title("Original Image")

plt.subplot(1, 3, 2)
plt.imshow(mask.squeeze(), cmap="gray")
plt.title("Predicted Mask")

plt.subplot(1, 3, 3)
plt.imshow(overlayed_image)
plt.title("Overlayed Segmentation")

plt.show()

# Save results
os.makedirs("output", exist_ok=True)
cv2.imwrite("output/predicted_mask.png", (mask.squeeze() * 255).astype(np.uint8))
cv2.imwrite("output/overlayed_segmentation.png", cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR))

print("✅ Segmentation results saved in 'output' folder!")
