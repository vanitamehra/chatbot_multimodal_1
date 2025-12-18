import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Current folder containing text files
txt_folder = "."  # current directory
# Folder to save PDFs
pdf_folder = "./pdfs"
os.makedirs(pdf_folder, exist_ok=True)

# PDF page settings
PAGE_WIDTH, PAGE_HEIGHT = letter
LEFT_MARGIN = 50
TOP_MARGIN = 750
LINE_HEIGHT = 20
FONT_SIZE = 12
FONT_NAME = "Helvetica"

def text_to_pdf(txt_path, pdf_path):
    """Convert a single text file to PDF with pagination."""
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")

    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFont(FONT_NAME, FONT_SIZE)
    
    y = TOP_MARGIN
    # Add title (file name without extension)
    title = os.path.splitext(os.path.basename(txt_path))[0].replace("_", " ").title()
    c.drawString(LEFT_MARGIN, y, title)
    y -= LINE_HEIGHT * 2

    for line in lines:
        if y < 50:  # bottom margin, create new page
            c.showPage()
            c.setFont(FONT_NAME, FONT_SIZE)
            y = TOP_MARGIN
        c.drawString(LEFT_MARGIN, y, line)
        y -= LINE_HEIGHT

    c.save()
    print(f"PDF created: {pdf_path}")

# Loop through all .txt files in the current folder
for filename in os.listdir(txt_folder):
    if filename.endswith(".txt"):
        txt_path = os.path.join(txt_folder, filename)
        pdf_filename = filename.replace(".txt", ".pdf")
        pdf_path = os.path.join(pdf_folder, pdf_filename)
        text_to_pdf(txt_path, pdf_path)

print("All text files converted to PDFs successfully!")
