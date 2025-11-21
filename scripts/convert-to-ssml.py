#!/usr/bin/env python3
"""
Convert notebook markdown files to SSML using OpenRouter API.
This script processes all markdown files in the notebook directory and converts them
to podcast-ready SSML format.
"""

import os
import sys
from pathlib import Path
import json
import requests
from datetime import datetime

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("Error: OPENROUTER_API_KEY environment variable not set")
    sys.exit(1)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-4o-mini"  # Affordable, high-quality model for SSML conversion

NOTEBOOK_DIR = Path(__file__).parent.parent / "notebook"
OUTPUT_DIR = Path(__file__).parent.parent / "podcast-ssml"

# SSML conversion prompt
SSML_CONVERSION_PROMPT = """You are a podcast script writer. Convert the following markdown technical content into SSML (Speech Synthesis Markup Language) format suitable for text-to-speech synthesis.

Guidelines:
1. Use proper SSML tags for natural speech:
   - <break time="500ms"/> for pauses between sections
   - <emphasis level="moderate"> for important terms
   - <prosody rate="slow"> for complex technical explanations
   - <say-as interpret-as="characters"> for acronyms like ASR, STT, etc.
2. Make the content conversational and engaging, as if explaining to a podcast listener
3. Remove markdown formatting (##, **, etc.) and convert to natural speech
4. Add appropriate pauses and pacing for clarity
5. For code snippets or technical terms, explain them verbally
6. Structure as a cohesive podcast segment with intro and outro phrases where appropriate
7. Maintain technical accuracy while making it accessible
8. The output should be wrapped in <speak> tags

Here's the content to convert:

{content}

Output only the SSML, nothing else."""


def read_markdown_file(filepath):
    """Read markdown file content."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def convert_to_ssml(content, filename):
    """Convert markdown content to SSML using OpenRouter."""
    print(f"Converting {filename} to SSML...")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://danielrosehill.com",
        "X-Title": "STT Fine-Tuning Podcast Generator"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": SSML_CONVERSION_PROMPT.format(content=content)
            }
        ]
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()

        result = response.json()
        ssml_content = result['choices'][0]['message']['content']

        return ssml_content

    except requests.exceptions.RequestException as e:
        print(f"Error converting {filename}: {e}")
        return None


def process_notebook():
    """Process all markdown files in the notebook directory."""
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Get all markdown files
    md_files = sorted(NOTEBOOK_DIR.rglob("*.md"))

    print(f"Found {len(md_files)} markdown files to process")

    # Create a manifest to track conversions
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "total_files": len(md_files),
        "files": []
    }

    for i, md_file in enumerate(md_files, 1):
        # Get relative path for organizing output
        rel_path = md_file.relative_to(NOTEBOOK_DIR)

        # Create corresponding output directory structure
        output_subdir = OUTPUT_DIR / rel_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Output SSML file path
        ssml_file = output_subdir / f"{md_file.stem}.ssml"

        print(f"\n[{i}/{len(md_files)}] Processing: {rel_path}")

        # Read markdown content
        try:
            content = read_markdown_file(md_file)
        except Exception as e:
            print(f"Error reading {md_file}: {e}")
            continue

        # Convert to SSML
        ssml_content = convert_to_ssml(content, rel_path)

        if ssml_content:
            # Save SSML file
            try:
                with open(ssml_file, 'w', encoding='utf-8') as f:
                    f.write(ssml_content)
                print(f"✓ Saved to: {ssml_file.relative_to(OUTPUT_DIR)}")

                manifest["files"].append({
                    "source": str(rel_path),
                    "output": str(ssml_file.relative_to(OUTPUT_DIR)),
                    "status": "success"
                })
            except Exception as e:
                print(f"Error saving {ssml_file}: {e}")
                manifest["files"].append({
                    "source": str(rel_path),
                    "output": str(ssml_file.relative_to(OUTPUT_DIR)),
                    "status": "error",
                    "error": str(e)
                })
        else:
            manifest["files"].append({
                "source": str(rel_path),
                "status": "conversion_failed"
            })

    # Save manifest
    manifest_file = OUTPUT_DIR / "manifest.json"
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Conversion complete!")
    print(f"✓ SSML files saved to: {OUTPUT_DIR}")
    print(f"✓ Manifest saved to: {manifest_file}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("STT Fine-Tuning Notebook → SSML Converter")
    print("=" * 60)
    print()

    process_notebook()


if __name__ == "__main__":
    main()
