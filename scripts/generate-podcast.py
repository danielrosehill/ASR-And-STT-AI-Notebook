#!/usr/bin/env python3
"""
Generate podcast audio from SSML files using TTS.
This script processes SSML files and converts them to audio using a natural-sounding TTS engine.
"""

import os
import sys
from pathlib import Path
import json
import subprocess
from datetime import datetime

# Configuration
SSML_DIR = Path(__file__).parent.parent / "podcast-ssml"
OUTPUT_DIR = Path(__file__).parent.parent / "podcast-audio"

# TTS Configuration - Using edge-tts (Microsoft Edge TTS) as a free, high-quality option
# Alternative options: Google Cloud TTS, AWS Polly, Azure TTS, or local solutions like Coqui TTS
DEFAULT_VOICE = "en-US-AriaNeural"  # Female voice, good for educational content
# Other options: en-US-GuyNeural (male), en-US-JennyNeural (female), en-GB-SoniaNeural (British)

def check_edge_tts():
    """Check if edge-tts is installed."""
    try:
        subprocess.run(["edge-tts", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_edge_tts():
    """Install edge-tts if not available."""
    print("edge-tts not found. Installing...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "edge-tts"], check=True)
        print("✓ edge-tts installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing edge-tts: {e}")
        return False


def convert_ssml_to_audio(ssml_file, output_file, voice=DEFAULT_VOICE):
    """Convert SSML file to audio using edge-tts."""
    print(f"Converting {ssml_file.name} to audio...")

    try:
        # edge-tts doesn't directly support SSML files, so we need to read content
        with open(ssml_file, 'r', encoding='utf-8') as f:
            ssml_content = f.read()

        # edge-tts command
        cmd = [
            "edge-tts",
            "--voice", voice,
            "--text", ssml_content,
            "--write-media", str(output_file)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print(f"✓ Audio saved to: {output_file}")
            return True
        else:
            print(f"Error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"Timeout converting {ssml_file.name}")
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


def process_ssml_directory(voice=DEFAULT_VOICE, concatenate=True):
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
        "voice": voice,
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

        # Output audio file path
        audio_file = output_subdir / f"{ssml_file.stem}.mp3"

        print(f"\n[{i}/{len(ssml_files)}] Processing: {rel_path}")

        # Convert to audio
        success = convert_ssml_to_audio(ssml_file, audio_file, voice)

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

        full_podcast_file = OUTPUT_DIR / f"stt-finetune-podcast-{datetime.now().strftime('%Y%m%d')}.mp3"
        if concatenate_audio_files(audio_files, full_podcast_file):
            # Get file size and duration
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


def list_available_voices():
    """List available voices from edge-tts."""
    print("Fetching available voices...")
    try:
        result = subprocess.run(["edge-tts", "--list-voices"], capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error listing voices: {e}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate podcast audio from SSML files")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="TTS voice to use")
    parser.add_argument("--list-voices", action="store_true", help="List available voices")
    parser.add_argument("--no-concatenate", action="store_true", help="Don't concatenate into single file")

    args = parser.parse_args()

    print("=" * 60)
    print("STT Fine-Tuning Podcast Generator - Stage 2: TTS")
    print("=" * 60)
    print()

    if args.list_voices:
        list_available_voices()
        return

    # Check if edge-tts is installed
    if not check_edge_tts():
        if not install_edge_tts():
            print("\nAlternatively, you can install it manually:")
            print("  pip install edge-tts")
            sys.exit(1)

    # Check if ffmpeg is available for concatenation
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: ffmpeg not found. Audio concatenation will not be available.")
        print("Install ffmpeg to enable full podcast generation:")
        print("  sudo apt install ffmpeg")
        args.no_concatenate = True

    process_ssml_directory(voice=args.voice, concatenate=not args.no_concatenate)


if __name__ == "__main__":
    main()
