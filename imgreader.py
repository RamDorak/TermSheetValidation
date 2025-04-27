import cv2
import pytesseract
import pandas as pd
import numpy as np

# Set path to Tesseract (Windows only)
pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    """Loads and preprocesses the image for better OCR accuracy."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (5,5), 0)
    _, img_bin = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    return img, img_bin

def extract_table(image_path, output_csv):
    """Detects table structure, extracts cell data, and saves it as a CSV."""
    img, img_bin = preprocess_image(image_path)

    # Detect horizontal and vertical lines
    kernel_h = np.ones((1, 50), np.uint8)
    kernel_v = np.ones((50, 1), np.uint8)
    
    horizontal_lines = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel_h)
    vertical_lines = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel_v)

    # Combine detected lines
    table_structure = cv2.add(horizontal_lines, vertical_lines)

    # Find contours (table cells)
    contours, _ = cv2.findContours(table_structure, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours from top-left to bottom-right
    contours = sorted(contours, key=lambda x: (cv2.boundingRect(x)[1], cv2.boundingRect(x)[0]))

    cell_data = []
    row = []
    last_y = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Extract each cell
        cell_img = img[y:y+h, x:x+w]
        text = pytesseract.image_to_string(cell_img, config="--psm 6").strip()

        if abs(y - last_y) > 10:  # New row detected
            if row:
                cell_data.append(row)
            row = []
            last_y = y

        row.append(text)

    if row:
        cell_data.append(row)

    # Convert to DataFrame & save as CSV
    df = pd.DataFrame(cell_data)
    df.to_csv(output_csv, index=False, header=False)
    print(f"Table successfully extracted and saved as {output_csv}")

# Example usage
extract_table("image.jpg", "output.csv")