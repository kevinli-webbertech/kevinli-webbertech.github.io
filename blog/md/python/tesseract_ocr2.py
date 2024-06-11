import cv2
import os
import argparse
import pytesseract
from PIL import Image

# Construct an Argument Parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image file")
ap.add_argument("-p", "--pre_processor", default="thresh", help="The preprocessor usage")
args = vars(ap.parse_args())

# Read the image with text
image = cv2.imread(args["image"])

# Convert to grayscale image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Check whether to apply threshold or blur
if args["pre_processor"] == "thresh":
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
elif args["pre_processor"] == "blur":
    gray = cv2.medianBlur(gray, 3)

# Save the processed image to disk
processed_image_path = "processed_image.jpg"
cv2.imwrite(processed_image_path, gray)

# Perform OCR on the processed image
text = pytesseract.image_to_string(Image.open(processed_image_path))
print(text)

# Save the original and processed images to disk
original_image_path = "original_image.jpg"
cv2.imwrite(original_image_path, image)

print(f"Original image saved to: {original_image_path}")
print(f"Processed image saved to: {processed_image_path}")
