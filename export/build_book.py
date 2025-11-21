#!/usr/bin/env python3
"""
Build PDF book from notebook markdown files
Organizes content by topic with table of contents
"""

import os
from pathlib import Path
from typing import Dict, List
import re

# Define topic organization and display names
TOPIC_ORDER = {
    'background-context': {
        'title': 'Part I: Background & Context',
        'description': 'Historical context and evolution of ASR technology'
    },
    'models': {
        'title': 'Part II: ASR Models',
        'description': 'Overview and comparison of ASR models'
    },
    'data-preparation': {
        'title': 'Part III: Data Preparation',
        'description': 'Audio data preparation and dataset creation'
    },
    'fine-tuning': {
        'title': 'Part IV: Fine-Tuning',
        'description': 'Fine-tuning strategies and techniques'
    },
    'inference': {
        'title': 'Part V: Inference & Deployment',
        'description': 'Running and deploying ASR models'
    },
    'amd': {
        'title': 'Part VI: AMD GPU Optimization',
        'description': 'AMD-specific hardware considerations'
    },
    'mobile-asr': {
        'title': 'Part VII: Mobile ASR',
        'description': 'Mobile and edge device deployment'
    },
    'formats': {
        'title': 'Part VIII: File Formats',
        'description': 'Audio and model file formats'
    },
    'vocab': {
        'title': 'Part IX: Vocabulary & Language',
        'description': 'Vocabulary recognition and language considerations'
    },
    'pitfalls': {
        'title': 'Part X: Common Pitfalls',
        'description': 'Common issues and how to avoid them'
    },
    'q-and-a': {
        'title': 'Part XI: Q&A',
        'description': 'Frequently asked questions'
    },
    'notes': {
        'title': 'Part XII: Additional Notes',
        'description': 'Supplementary topics and observations'
    }
}


def clean_title(filename: str) -> str:
    """Convert filename to readable title"""
    # Remove .md extension and convert hyphens to spaces
    title = filename.replace('.md', '').replace('-', ' ')
    # Capitalize each word
    return ' '.join(word.capitalize() for word in title.split())


def collect_files() -> Dict[str, List[Path]]:
    """Collect all markdown files organized by topic"""
    notebook_dir = Path('../notebook')
    organized = {topic: [] for topic in TOPIC_ORDER.keys()}

    for topic_dir in notebook_dir.iterdir():
        if topic_dir.is_dir() and topic_dir.name in TOPIC_ORDER:
            md_files = sorted(topic_dir.glob('*.md'))
            organized[topic_dir.name] = md_files

    return organized


def generate_toc(organized_files: Dict[str, List[Path]]) -> str:
    """Generate compact table of contents - parts only"""
    toc = ["# Speech-to-Text Fine-Tuning Guide", "",
           "_A Comprehensive Guide to ASR Model Fine-Tuning and Deployment_", "",
           "---", "", "## Table of Contents", ""]

    for topic_key in TOPIC_ORDER.keys():
        if organized_files.get(topic_key):
            topic_info = TOPIC_ORDER[topic_key]
            chapter_count = len(organized_files[topic_key])
            toc.append(f"**{topic_info['title']}**  ")
            toc.append(f"{topic_info['description']} ({chapter_count} chapters)")
            toc.append("")

    toc.append("---")
    toc.append("")

    return '\n'.join(toc)


def read_file_content(file_path: Path) -> str:
    """Read and return file content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content.strip()
    except Exception as e:
        return f"<!-- Error reading {file_path}: {e} -->"


def build_master_document(organized_files: Dict[str, List[Path]]) -> str:
    """Build the complete master document"""
    parts = []

    # Add TOC
    parts.append(generate_toc(organized_files))

    # Add content by topic
    for topic_key in TOPIC_ORDER.keys():
        if not organized_files.get(topic_key):
            continue

        topic_info = TOPIC_ORDER[topic_key]

        # Topic section header (h1 - will get page break from CSS)
        parts.append("")
        parts.append(f"# {topic_info['title']}")
        parts.append("")
        parts.append(f"_{topic_info['description']}_")
        parts.append("")
        parts.append("---")
        parts.append("")

        # Add each chapter in this topic
        for file_path in organized_files[topic_key]:
            chapter_title = clean_title(file_path.name)

            # Chapter header (h2 - no page break)
            parts.append("")
            parts.append(f"## {chapter_title}")
            parts.append("")

            # Chapter content
            content = read_file_content(file_path)

            # Remove any existing top-level headers to avoid conflicts
            content = re.sub(r'^#\s+.*$', '', content, flags=re.MULTILINE)

            parts.append(content)
            parts.append("")

    return '\n'.join(parts)


def main():
    """Main execution"""
    print("Building Speech-to-Text Fine-Tuning Guide...")

    # Collect files
    print("Collecting markdown files...")
    organized_files = collect_files()

    total_files = sum(len(files) for files in organized_files.values())
    print(f"Found {total_files} files across {len(TOPIC_ORDER)} topics")

    # Build master document
    print("Building master document...")
    master_content = build_master_document(organized_files)

    # Write master markdown
    output_md = Path('STT-Fine-Tuning-Guide.md')
    print(f"Writing to {output_md}...")
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write(master_content)

    print(f"✓ Master document created: {output_md}")
    print(f"  Total size: {len(master_content):,} characters")
    print("\nNext: Convert to PDF using pandoc")


if __name__ == '__main__':
    main()
