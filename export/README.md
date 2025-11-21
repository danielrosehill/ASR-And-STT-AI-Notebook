# STT Fine-Tuning Guide - Export

This directory contains the exported PDF book version of the Speech-to-Text Fine-Tuning notebook.

## Files

- **STT-Fine-Tuning-Guide.pdf** - The complete book in PDF format with professional styling
- **STT-Fine-Tuning-Guide.md** - Master markdown document with all content concatenated
- **STT-Fine-Tuning-Guide.html** - HTML version with styling
- **build_book.py** - Python script that organizes and concatenates all notebook content
- **book-style.css** - Custom CSS for professional PDF styling

## Book Features

✓ **Complete Table of Contents** - All 41 chapters organized into 12 thematic parts
✓ **Page Numbers** - Footer displays page numbers on every page (except title page)
✓ **Professional Typography** - Georgia serif font for readability, Helvetica for headings
✓ **Organized by Topic** - Content structured in logical progression:
  - Part I: Background & Context
  - Part II: ASR Models
  - Part III: Data Preparation
  - Part IV: Fine-Tuning
  - Part V: Inference & Deployment
  - Part VI: AMD GPU Optimization
  - Part VII: Mobile ASR
  - Part VIII: File Formats
  - Part IX: Vocabulary & Language
  - Part X: Common Pitfalls
  - Part XI: Q&A
  - Part XII: Additional Notes

✓ **Page Breaks** - Clean separation between chapters and sections
✓ **Readable Layout** - Justified text, proper margins (2.5cm top, 2cm sides, 3cm bottom)
✓ **Code Highlighting** - Syntax-friendly formatting for code blocks
✓ **Footer Metadata** - Document title in footer for easy reference

## Regenerating the PDF

To rebuild the PDF with updated content:

```bash
# From the export directory
python3 build_book.py
pandoc STT-Fine-Tuning-Guide.md -o STT-Fine-Tuning-Guide.html \
  --standalone --toc --toc-depth=2 --css=book-style.css \
  --metadata title="Speech-to-Text Fine-Tuning Guide" \
  --metadata author="Daniel Rosehill"
weasyprint STT-Fine-Tuning-Guide.html STT-Fine-Tuning-Guide.pdf
```

## Statistics

- **Total Chapters**: 41
- **Total Topics**: 12
- **Document Size**: ~490,000 characters
- **PDF Size**: 1.2 MB
- **Format**: A4, professional book layout

---

Generated using pandoc and WeasyPrint with custom CSS styling.
