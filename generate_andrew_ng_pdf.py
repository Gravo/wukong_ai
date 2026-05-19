#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate PDF biography for Andrew Ng using reportlab + msyh.ttc
Standard 12-chapter structure with professional formatting
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import re
import os

# Register Chinese font
font_path = "C:/Windows/Fonts/msyh.ttc"
if os.path.exists(font_path):
    pdfmetrics.registerFont(TTFont('MSYH', font_path))
    pdfmetrics.registerFont(TTFont('MSYH-Bold', font_path))
    chinese_font = 'MSYH'
    chinese_font_bold = 'MSYH-Bold'
else:
    print("Warning: msyh.ttc not found, using default font")
    chinese_font = 'Helvetica'
    chinese_font_bold = 'Helvetica-Bold'

# Create output directory
output_dir = "C:/Users/Gao Wei/.qclaw/workspace/andrew-ng-book"
os.makedirs(output_dir, exist_ok=True)

# Create PDF
pdf_path = f"{output_dir}/andrew-ng-book.pdf"
doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                        rightMargin=72, leftMargin=72,
                        topMargin=72, bottomMargin=18)

# Define styles
styles = getSampleStyleSheet()

# Title style
title_style = ParagraphStyle(
    'ChineseTitle',
    parent=styles['Heading1'],
    fontName=chinese_font_bold,
    fontSize=24,
    alignment=TA_CENTER,
    spaceAfter=30,
)

# Chapter style
chapter_style = ParagraphStyle(
    'ChineseChapter',
    parent=styles['Heading2'],
    fontName=chinese_font_bold,
    fontSize=18,
    spaceAfter=12,
    spaceBefore=12,
)

# Section style
section_style = ParagraphStyle(
    'ChineseSection',
    parent=styles['Heading3'],
    fontName=chinese_font_bold,
    fontSize=14,
    spaceAfter=10,
    spaceBefore=10,
)

# Body style
body_style = ParagraphStyle(
    'ChineseBody',
    parent=styles['BodyText'],
    fontName=chinese_font,
    fontSize=11,
    leading=16,
    alignment=TA_JUSTIFY,
    spaceAfter=8,
)

# Read Markdown file
md_path = "C:/Users/Gao Wei/.qclaw/workspace/andrew-ng-book/andrew-ng-biography.md"
with open(md_path, 'r', encoding='utf-8') as f:
    md_content = f.read()

# Parse Markdown and convert to PDF elements
story = []

lines = md_content.split('\n')
in_code_block = False
current_para = []

for line in lines:
    # Skip code blocks
    if line.strip().startswith('```'):
        in_code_block = not in_code_block
        continue
    
    if in_code_block:
        continue
    
    # Title
    if line.strip() == '# 吴恩达传：人工智能教育的布道者与产业实践的先驱':
        story.append(Paragraph(line.strip('# ').strip(), title_style))
        story.append(Spacer(1, 0.2*inch))
        continue
    
    # Subtitle
    if line.strip().startswith('**') and line.strip().endswith('**'):
        text = line.strip('*')
        if '第二十本' in text:
            story.append(Paragraph(text, chapter_style))
            story.append(Spacer(1, 0.3*inch))
            continue
    
    # Chapter (## )
    if re.match(r'^## 第[一二三四五六七八九十]+章', line):
        text = line.strip('#').strip()
        story.append(PageBreak())
        story.append(Paragraph(text, chapter_style))
        story.append(Spacer(1, 0.2*inch))
        continue
    
    # Section (### )
    if re.match(r'^### \d+\.\d+', line):
        text = line.strip('#').strip()
        story.append(Paragraph(text, section_style))
        continue
    
    # Bullet list
    if line.strip().startswith('- '):
        text = line.strip('- ')
        story.append(Paragraph(f"• {text}", body_style))
        continue
    
    # Table (simplified - convert to paragraph)
    if '|' in line and '---' not in line:
        # Skip table for now, convert to paragraph
        cells = [c.strip() for c in line.split('|') if c.strip()]
        if cells:
            text = ' | '.join(cells)
            story.append(Paragraph(text, body_style))
        continue
    
    # Regular paragraph
    if line.strip() and not line.strip().startswith('#'):
        # Clean up markdown formatting
        text = line.strip()
        text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', text)  # Italic
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # Links
        
        if text:
            try:
                story.append(Paragraph(text, body_style))
            except:
                # Fallback for problematic text
                story.append(Paragraph(text.encode('latin-1', errors='ignore').decode('latin-1'), body_style))
    elif not line.strip():
        # Empty line - add spacer
        if current_para:
            story.append(Spacer(1, 0.1*inch))
            current_para = []

# Build PDF
doc.build(story)
print(f"PDF generated successfully: {pdf_path}")
print(f"File size: {os.path.getsize(pdf_path)} bytes")
