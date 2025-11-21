#!/usr/bin/env python3
"""
Generate podcast audio from SSML files using Kokoro-82M TTS from Hugging Face.
This script processes SSML files and converts them to audio using the Kokoro TTS model.
"""

import os
import sys
from pathlib import Path
import json
import subprocess
from datetime import datetime
import requests
import re

# Configuration
SSML_DIR = Path(__file__).parent.parent / "podcast-ssml"
OUTPUT_DIR = Path(__file__).parent.parent / "podcast-audio"

# Hugging Face configuration
HF_API_KEY = os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY")
if not HF_API_KEY:
    print("Error: HF_TOKEN or HF_API_KEY environment variable not set")
    print("Get your token from: https://huggingface.co/settings/tokens")
    sys.exit(1)

KOKORO_MODEL = "hexgrad/Kokoro-82M"
HF_API_URL = f"https://api-inference.huggingface.co/models/{KOKORO_MODEL}"

def strip_ssml_tags(ssml_content):
    """Remove SSML tags and return plain text for TTS."""
    # Remove <?xml ... ?> declaration
    text = re.sub(r'<\?xml[^>]*\?>', '', ssml_content)

    # Remove <speak> tags
    text = re.sub(r'<speak[^>]*>', '', text)
    text = re.sub(r'</speak>', '', text)

    # Remove break tags
    text = re.sub(r'<break[^>]*/?>', ' ', text)

    # Remove prosody tags but keep content
    text = re.sub(r'<prosody[^>]*>', '', text)
    text = re.sub(r'</prosody>', '', text)

    # Remove emphasis tags but keep content
    text = re.sub(r'<emphasis[^>]*>', '', text)
    text = re.sub(r'</emphasis>', '', text)

    # Remove say-as tags but keep content
    text = re.sub(r'<say-as[^>]*>', '', text)
    text = re.sub(r'</say-as>', '', text)

    # Remove paragraph tags but keep content
    text = re.sub(r'<p>', '\n\n', text)
    text = re.sub(r'</p>', '\n\n', text)

    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text


def convert_ssml_to_audio_kokoro(ssml_file, output_file):
    """Convert SSML file to audio using Kokoro-82M via Hugging Face API."""
    print(f"Converting {ssml_file.name} to audio...")

    try:
        # Read SSML content
        with open(ssml_file, 'r', encoding='utf-8') as f:
            ssml_content = f.read()

        # Strip SSML tags to get plain text
        text = strip_ssml_tags(ssml_content)

        if not text:
            print(f"Warning: No text content found in {ssml_file.name}")
            return False

        # Call Hugging Face Inference API
        headers = {
            "Authorization": f"Bearer {HF_API_KEY}"
        }

        payload = {
            "inputs": text
        }

        response = requests.post(
            HF_API_URL,
            headers=headers,
            json=payload,
            timeout=300
        )

        if response.status_code == 200:
            # Save audio file
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"✓ Audio saved to: {output_file}")
            return True
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        print(f"Error converting {ssml_file.name}: {e}")
        return False


def concatenate_audio_files(audio_files, output_file):
    """Concatenate multiple audio files into a single podcast episode."""
    print(f"\nConcatenating {len(audio_files)} audio files...")

    # Create a temporary file list for ffmpeg
    file_list_path = OUTPUT_DIR / "concat_list.txt"

    try:
        with open(file_list_path, 'w', encoding='utf-8') as f:
            for audio_file in audio_files:
                # FFmpeg concat requires relative or absolute paths in specific format
                f.write(f"file '{audio_file.absolute()}'\n")

        # Use ffmpeg to concatenate
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", str(file_list_path),
            "-c", "copy",
            "-y",  # Overwrite output file
            str(output_file)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✓ Podcast saved to: {output_file}")
            return True
        else:
            print(f"Error concatenating audio: {result.stderr}")
            return False

    except Exception as e:
        print(f"Error concatenating audio: {e}")
        return False
    finally:
        # Clean up temporary file list
        if file_list_path.exists():
            file_list_path.unlink()


def process_ssml_directory(concatenate=True):
    """Process all SSML files in the directory."""
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load manifest from SSML conversion
    manifest_file = SSML_DIR / "manifest.json"
    if not manifest_file.exists():
        print("Error: manifest.json not found. Run convert-to-ssml.py first.")
        sys.exit(1)

    with open(manifest_file, 'r') as f:
        manifest = json.load(f)

    # Get all SSML files
    ssml_files = sorted(SSML_DIR.rglob("*.ssml"))

    if not ssml_files:
        print("Error: No SSML files found. Run convert-to-ssml.py first.")
        sys.exit(1)

    print(f"Found {len(ssml_files)} SSML files to process")

    # Create podcast manifest
    podcast_manifest = {
        "generated_at": datetime.now().isoformat(),
        "tts_engine": "kokoro-82m",
        "model": KOKORO_MODEL,
        "total_files": len(ssml_files),
        "files": []
    }

    audio_files = []

    for i, ssml_file in enumerate(ssml_files, 1):
        # Get relative path for organizing output
        rel_path = ssml_file.relative_to(SSML_DIR)

        # Create corresponding output directory structure
        output_subdir = OUTPUT_DIR / rel_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Output audio file path (Kokoro outputs WAV by default)
        audio_file = output_subdir / f"{ssml_file.stem}.wav"

        print(f"\n[{i}/{len(ssml_files)}] Processing: {rel_path}")

        # Convert to audio
        success = convert_ssml_to_audio_kokoro(ssml_file, audio_file)

        if success:
            audio_files.append(audio_file)
            podcast_manifest["files"].append({
                "ssml_source": str(rel_path),
                "audio_output": str(audio_file.relative_to(OUTPUT_DIR)),
                "status": "success"
            })
        else:
            podcast_manifest["files"].append({
                "ssml_source": str(rel_path),
                "status": "failed"
            })

    # Save individual files manifest
    individual_manifest_file = OUTPUT_DIR / "individual-files-manifest.json"
    with open(individual_manifest_file, 'w', encoding='utf-8') as f:
        json.dump(podcast_manifest, f, indent=2)

    print(f"\n✓ Individual audio files generated!")
    print(f"✓ Manifest saved to: {individual_manifest_file}")

    # Optionally concatenate all audio files into a single podcast
    if concatenate and audio_files:
        print("\n" + "=" * 60)
        print("Concatenating audio files into full podcast...")
        print("=" * 60)

        full_podcast_file = OUTPUT_DIR / f"stt-finetune-podcast-{datetime.now().strftime('%Y%m%d')}.wav"
        if concatenate_audio_files(audio_files, full_podcast_file):
            # Get file size
            file_size_mb = full_podcast_file.stat().st_size / (1024 * 1024)

            # Update manifest
            podcast_manifest["full_podcast"] = {
                "file": str(full_podcast_file.relative_to(OUTPUT_DIR)),
                "size_mb": round(file_size_mb, 2),
                "track_count": len(audio_files)
            }

            # Save final manifest
            final_manifest_file = OUTPUT_DIR / "podcast-manifest.json"
            with open(final_manifest_file, 'w', encoding='utf-8') as f:
                json.dump(podcast_manifest, f, indent=2)

            print(f"\n✓ Full podcast created!")
            print(f"  File: {full_podcast_file}")
            print(f"  Size: {file_size_mb:.2f} MB")
            print(f"  Tracks: {len(audio_files)}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate podcast audio from SSML using Kokoro-82M")
    parser.add_argument("--no-concatenate", action="store_true", help="Don't concatenate into single file")

    args = parser.parse_args()

    print("=" * 60)
    print("STT Fine-Tuning Podcast Generator - Kokoro-82M TTS")
    print("=" * 60)
    print()

    # Check if ffmpeg is available for concatenation
    if not args.no_concatenate:
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Warning: ffmpeg not found. Audio concatenation will not be available.")
            print("Install ffmpeg to enable full podcast generation:")
            print("  sudo apt install ffmpeg")
            args.no_concatenate = True

    process_ssml_directory(concatenate=not args.no_concatenate)


if __name__ == "__main__":
    main()
