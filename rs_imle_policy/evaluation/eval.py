import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


# Function to calculate IoU and coverage
def get_mask_metrics(target_mask, mask):
    total = np.sum(target_mask)
    i = np.sum(target_mask & mask)
    u = np.sum(target_mask | mask)
    iou = i / u
    coverage = i / total
    result = {"iou": iou, "coverage": coverage}
    return result


# Function to load and crop image as a binary mask, with optional red masking
def load_image_as_mask(file_path, crop_box=None, threshold=128, mask_red=False):
    """
    Loads an image from file, applies cropping if specified, converts it to grayscale,
    and optionally applies a mask for red regions.

    Parameters:
    - file_path: str, path to the image file.
    - crop_box: tuple, (left, upper, right, lower) crop coordinates.
    - threshold: int, threshold for binary mask creation.
    - mask_red: bool, if True applies red region masking.

    Returns:
    - mask: numpy array, binary mask.
    - image: PIL Image, grayscale (and cropped) image.
    """
    # Open and optionally crop the image
    image = Image.open(file_path).convert("RGB")
    if crop_box:
        image = image.crop(crop_box)
    image_np = np.array(image)

    # Apply red masking if specified
    if mask_red:
        mask = get_t_mask(image_np)  # Mask for red regions
    else:
        # Convert to grayscale and apply binary threshold
        gray_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY))
        mask = np.array(gray_image) > threshold

    return mask, image  # Return both the binary mask and processed image


# Function to generate a mask for red regions in the image
def get_t_mask(img, hsv_ranges=None):
    if hsv_ranges is None:
        hsv_ranges = [
            [0, 255],  # Hue range
            [85, 255],  # Saturation range
            [82, 255],  # Value range
        ]
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = np.ones(img.shape[:2], dtype=bool)
    for c in range(len(hsv_ranges)):
        low_bound, high_bound = hsv_ranges[c]
        mask &= (low_bound <= hsv_img[..., c]) & (hsv_img[..., c] <= high_bound)
    return mask


# Example usage
target_mask_path = "target_mask.png"
predicted_mask_path = "target_mask.png"

# Define crop box (left, upper, right, lower) as desired
crop_box = (130, 130, 450, 330)

# Load and crop masks and grayscale images, applying red masking
target_mask, target_image = load_image_as_mask(
    target_mask_path, crop_box=crop_box, mask_red=True
)
predicted_mask, predicted_image = load_image_as_mask(
    predicted_mask_path, crop_box=crop_box, mask_red=True
)

# Compute metrics
metrics = get_mask_metrics(target_mask, predicted_mask)
print("IoU:", metrics["iou"])
print("Coverage:", metrics["coverage"])

# Visualize in a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Display the cropped RGB target image
axes[0, 0].imshow(target_image)
axes[0, 0].set_title("Target Image (Cropped RGB)")
axes[0, 0].axis("off")

# Display the cropped RGB predicted image
axes[0, 1].imshow(predicted_image)
axes[0, 1].set_title("Predicted Image (Cropped RGB)")
axes[0, 1].axis("off")

# Display the binary target mask for red regions
axes[1, 0].imshow(target_mask, cmap="gray")
axes[1, 0].set_title("Target Mask (Red Mask)")
axes[1, 0].axis("off")

# Display the binary predicted mask for red regions
axes[1, 1].imshow(predicted_mask, cmap="gray")
axes[1, 1].set_title("Predicted Mask (Red Mask)")
axes[1, 1].axis("off")

plt.tight_layout()
plt.show()
