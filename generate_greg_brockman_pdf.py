#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Greg Brockman biography PDF with reportlab
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

# Register Chinese font
font_path = "C:\\Windows\\Fonts\\msyh.ttc"  # Microsoft YaHei
if os.path.exists(font_path):
    pdfmetrics.registerFont(TTFont('msyh', font_path))
    chinese_font = 'msyh'
else:
    print("Warning: msyh.ttc not found, using default font")
    chinese_font = 'Helvetica'

# Create PDF
pdf_path = "C:\\Users\\Gao Wei\\.qclaw\\workspace\\greg-brockman-book\\greg-brockman-biography.pdf"
doc = SimpleDocTemplate(pdf_path, pagesize=A4)
styles = getSampleStyleSheet()

# Create custom styles with Chinese font support
title_style = ParagraphStyle(
    'ChineseTitle',
    parent=styles['Heading1'],
    fontName=chinese_font,
    fontSize=18,
    leading=22,
    alignment=1,  # Center
)

heading1_style = ParagraphStyle(
    'ChineseHeading1',
    parent=styles['Heading1'],
    fontName=chinese_font,
    fontSize=14,
    leading=18,
)

heading2_style = ParagraphStyle(
    'ChineseHeading2',
    parent=styles['Heading2'],
    fontName=chinese_font,
    fontSize=12,
    leading=16,
)

body_style = ParagraphStyle(
    'ChineseBody',
    parent=styles['BodyText'],
    fontName=chinese_font,
    fontSize=10,
    leading=14,
)

# Read markdown content
md_path = "C:\\Users\\Gao Wei\\.qclaw\\workspace\\greg-brockman-book\\greg-brockman-biography.md"
with open(md_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Parse markdown and create PDF elements
story = []

lines = content.split('\n')
for line in lines:
    line = line.strip()
    if not line:
        continue
    
    if line.startswith('# ') and not line.startswith('## '):
        # Title
        text = line[2:].strip()
        story.append(Paragraph(text, title_style))
        story.append(Spacer(1, 0.2*inch))
    elif line.startswith('## '):
        # Heading 1
        text = line[3:].strip()
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(text, heading1_style))
        story.append(Spacer(1, 0.1*inch))
    elif line.startswith('### '):
        # Heading 2
        text = line[4:].strip()
        story.append(Paragraph(text, heading2_style))
        story.append(Spacer(1, 0.05*inch))
    elif line.startswith('- ') or line.startswith('* '):
        # List item
        text = line[2:].strip()
        story.append(Paragraph(f"• {text}", body_style))
    else:
        # Body text
        if line and not line.startswith('#') and not line.startswith('---'):
            story.append(Paragraph(line, body_style))
            story.append(Spacer(1, 0.05*inch))

# Build PDF
doc.build(story)
print(f"PDF generated: {pdf_path}")