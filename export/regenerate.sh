#!/bin/bash
# Regenerate the STT Fine-Tuning Guide PDF

cd "$(dirname "$0")"

echo "Building master markdown document..."
python3 build_book.py

echo "Converting to HTML..."
pandoc STT-Fine-Tuning-Guide.md -o STT-Fine-Tuning-Guide.html \
  --standalone --toc --toc-depth=1 --css=book-style.css \
  --metadata title="Speech-to-Text Fine-Tuning Guide" \
  --metadata author="Daniel Rosehill"

echo "Converting to PDF..."
weasyprint STT-Fine-Tuning-Guide.html STT-Fine-Tuning-Guide.pdf 2>&1 | grep -v "WARNING"

echo ""
echo "✓ PDF generated successfully!"
ls -lh STT-Fine-Tuning-Guide.pdf
pdfinfo STT-Fine-Tuning-Guide.pdf 2>/dev/null | grep "Pages:"
