import fitz  # PyMuPDF
import os

pdf_path = "assets/papers/Virtual-Rodent-Aldarondo-DeepMind.pdf"
output_dir = "assets/papers/Virtual-Rodent-Extracted"

# Create output directories
os.makedirs(f"{output_dir}/figures", exist_ok=True)

# Open PDF
doc = fitz.open(pdf_path)

# Extract text from all pages
full_text = []
for page_num, page in enumerate(doc):
    text = page.get_text()
    full_text.append(f"\n\n--- PAGE {page_num + 1} ---\n\n{text}")

# Save full text
with open(f"{output_dir}/full_text.txt", "w", encoding="utf-8") as f:
    f.write("".join(full_text))

# Extract images from each page
img_count = 0
for page_num, page in enumerate(doc):
    images = page.get_images(full=True)
    for img_idx, img in enumerate(images):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        
        img_filename = f"{output_dir}/figures/page{page_num+1}_img{img_idx+1}.{image_ext}"
        with open(img_filename, "wb") as f:
            f.write(image_bytes)
        img_count += 1

doc.close()
print(f"Extracted text to {output_dir}/full_text.txt")
print(f"Extracted {img_count} images to {output_dir}/figures/")