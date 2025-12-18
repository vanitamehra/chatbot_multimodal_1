from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

# Folder to save PDFs
pdf_folder = "backend/data/pdfs"
os.makedirs(pdf_folder, exist_ok=True)

# Example courses and curriculum
courses = {
    "data_science": [
        "Week 1-2: Python Programming and Data Analysis",
        "Week 3-4: Statistics and Probability",
        "Week 5-6: Data Visualization",
        "Week 7-8: Machine Learning Basics",
        "Week 9-10: Advanced ML",
        "Week 11: Capstone Project",
        "Week 12: Deployment and Presentation"
    ],
    "web_development": [
        "Week 1: HTML, CSS Basics",
        "Week 2: Advanced CSS, Flexbox, Grid",
        "Week 3: JavaScript Fundamentals",
        "Week 4: DOM Manipulation and Events",
        "Week 5: Frontend Frameworks (React Basics)",
        "Week 6: Backend Basics (Node.js, Express)",
        "Week 7: Database (MongoDB / SQL)",
        "Week 8: REST APIs",
        "Week 9: Project Work",
        "Week 10: Deployment (Netlify / Heroku)"
    ]
}

for course, curriculum in courses.items():
    pdf_path = os.path.join(pdf_folder, f"{course}.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(50, 750, f"{course.replace('_', ' ').title()} Curriculum")
    y = 720
    for week in curriculum:
        c.drawString(50, y, week)
        y -= 20
    c.save()
    print(f"PDF created: {pdf_path}")
