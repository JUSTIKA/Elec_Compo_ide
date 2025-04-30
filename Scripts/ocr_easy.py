import cv2
import easyocr
import re


# load the image
image = cv2.imread(r'C:\Users\vaioj\OneDrive\Documents\GitHub\Elec_Compo_ide\OCR\image_test_ocr_v2.jpeg')
image = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

# ocr
reader = easyocr.Reader(['en'], gpu=True)
results = reader.readtext(image)

# save the results
with open("results_easy.txt", "w", encoding="utf-8") as file:
    found = set()
    for (_, text, conf) in results:
        cleaned = text.strip()
        # we try to match the references
        if conf > 0.1 and re.match(r'^[CR]\d{1,3}$', cleaned) and cleaned not in found:
            file.write(cleaned + "\n")
            found.add(cleaned)

print("done")
