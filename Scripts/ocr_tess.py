import cv2
import pytesseract
import re

# load the image
image = cv2.imread(r'C:\Users\vaioj\OneDrive\Documents\GitHub\Elec_Compo_ide\OCR\image_test_ocr_v1.jpeg')
image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

# whitelist stuff
custom_config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=CR0123456789'

# bounding boxes
data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)

# draw the bounding boxes
for i in range(len(data['text'])):
    conf = int(data['conf'][i])
    text = data['text'][i].strip()
    
    if conf > 20 and text:  # confidence for detection
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Save debug results
cv2.imwrite("debug_output.png", image)
print("done debug'")

# Extract clean words
words = [data['text'][i].strip() for i in range(len(data['text']))
         if int(data['conf'][i]) > 30 and data['text'][i].strip()]

# Filter references
references = [w for w in words if re.match(r'^[CR]0*\d{1,3}$', w)]

# save results
with open("results.txt", "w", encoding="utf-8") as file:
    for ref in sorted(set(references)):
        file.write(ref + "\n")

print("done")