from PIL import Image, ImageDraw
from IPython.display import display
import numpy as np
import cv2
import os

#Script found on https://www.kaggle.com/code/farahalarbeed/convert-binary-masks-to-yolo-format/notebook

def process_mask(mask_path):
    print(f"Processing mask: {mask_path}")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Unable to read the file at {mask_path}. Please check the file path or integrity.")

    # Image processing
    _, binary_mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)  # Adjust threshold value
    #display(Image.fromarray(binary_mask))  # Visualize the binary mask
    #Image.fromarray(binary_mask).show()  # Opens the binary mask in the default image viewer
    #Image.fromarray(binary_mask).save("binary_mask.png")  # Saves the binary mask to a file
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Number of contours found: {len(contours)}")

    objects_info = []

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        # Filter out contours that are too large or too small
        if width < 10 or height < 10:  # Only filter out very small contours
            print(f"Ignoring small contour: x={x}, y={y}, width={width}, height={height}")
            continue

        print(f"Bounding box: x={x}, y={y}, width={width}, height={height}")

        class_label = 6 
        x_center, y_center, normalized_width, normalized_height = convert_coordinates_to_yolo(mask.shape[1], mask.shape[0], x, y, width, height)
        print(f"YOLO coordinates: x_center={x_center}, y_center={y_center}, width={normalized_width}, height={normalized_height}")
        objects_info.append((class_label, x_center, y_center, normalized_width, normalized_height))

    print(f"Objects info to write: {objects_info}")
    return objects_info

def convert_coordinates_to_yolo(image_width, image_height, x, y, width, height):
    x_center = (x + width / 2) / image_width
    y_center = (y + height / 2) / image_height
    normalized_width = width / image_width
    normalized_height = height / image_height

    return x_center, y_center, normalized_width, normalized_height

def write_yolo_annotations(output_path, image_name, objects_info):
    annotation_file_path = os.path.join(output_path, image_name)
    print(f"Writing annotations to: {annotation_file_path}")
    print(f"Objects info to write: {objects_info}")

    with open(annotation_file_path, "w") as file:
        for obj_info in objects_info:
            line = f"{obj_info[0]} {obj_info[1]} {obj_info[2]} {obj_info[3]} {obj_info[4]}\n"
            print(f"Writing line: {line.strip()}")
            file.write(line)

input_path = r"C:\Users\vaioj\OneDrive\Documents\GitHub\Elec_Compo_ide\wire_seg\masks"
output_path = r"C:\Users\vaioj\OneDrive\Documents\GitHub\Elec_Compo_ide\wire_seg\labels"
imgs_path = r"C:\Users\vaioj\OneDrive\Documents\GitHub\Elec_Compo_ide\wire_seg\imgs"

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

def read_yolo_annotations(annotation_path):
    with open(annotation_path, "r") as file:
        lines = file.readlines()
        objects_info = [list(map(float, line.strip().split())) for line in lines]
    return objects_info

def preprocess_for_yolo(image_path, annotation_path):
    image = Image.open(image_path)
    width, height = image.size
    draw = ImageDraw.Draw(image)
    objects_info = read_yolo_annotations(annotation_path)

    for obj_info in objects_info:
        label, x_center, y_center, normalized_width, normalized_height = obj_info

        # Convert YOLO coordinates to pixel coordinates
        x, y, w, h = int((x_center - normalized_width / 2) * width), int((y_center - normalized_height / 2) * height), int(normalized_width * width), int(normalized_height * height)

        # Draw rectangle on the image
        draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=2)

    display(image)

# Process each image in the imgs directory
for img_name in os.listdir(imgs_path):
    if img_name.endswith(".jpg"):  # Process only .jpg files
        # Get the corresponding mask file
        mask_name = os.path.splitext(img_name)[0] + ".jpg"
        mask_path = os.path.join(input_path, mask_name)

        # Ensure the mask file exists
        if not os.path.exists(mask_path):
            print(f"Mask not found for image: {img_name}")
            continue

        # Generate the annotation file name
        annotation_name = os.path.splitext(img_name)[0] + ".txt"

        # Process the mask and write annotations
        objects_info = process_mask(mask_path)
        write_yolo_annotations(output_path, annotation_name, objects_info)

        # Visualize the annotations on the image
        img_path = os.path.join(imgs_path, img_name)
        annotation_path = os.path.join(output_path, annotation_name)
        preprocess_for_yolo(img_path, annotation_path)



image_path = r'C:\Users\vaioj\OneDrive\Documents\GitHub\Elec_Compo_ide\wire_seg\imgs\c1_0.jpg'
annotation_path = r'C:\Users\vaioj\OneDrive\Documents\GitHub\Elec_Compo_ide\wire_seg\labels\0.txt'

preprocess_for_yolo(image_path, annotation_path)