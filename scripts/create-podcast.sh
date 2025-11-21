#!/bin/bash
# Master script to create podcast from notebook

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "STT Fine-Tuning Notebook Podcast Creator"
echo "========================================"
echo ""

# Check for required environment variables
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Error: OPENROUTER_API_KEY environment variable not set"
    echo "Please set it with: export OPENROUTER_API_KEY='your-key-here'"
    exit 1
fi

# Stage 1: Convert to SSML
echo "Stage 1: Converting notebook to SSML using OpenRouter..."
echo "=========================================="
python3 "$SCRIPT_DIR/convert-to-ssml.py"

if [ $? -ne 0 ]; then
    echo "Error in Stage 1"
    exit 1
fi

echo ""
echo "Stage 1 complete!"
echo ""
read -p "Proceed to Stage 2 (TTS generation)? [Y/n] " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ -n $REPLY ]]; then
    echo "Stopping after Stage 1. SSML files are in: podcast-ssml/"
    exit 0
fi

# Stage 2: Generate audio
echo ""
echo "Stage 2: Generating podcast audio from SSML..."
echo "=========================================="
python3 "$SCRIPT_DIR/generate-podcast.py"

if [ $? -ne 0 ]; then
    echo "Error in Stage 2"
    exit 1
fi

echo ""
echo "========================================"
echo "Podcast generation complete!"
echo "========================================"
echo ""
echo "Output locations:"
echo "  SSML files: $PROJECT_DIR/podcast-ssml/"
echo "  Audio files: $PROJECT_DIR/podcast-audio/"
echo ""

# Show the final podcast file if it exists
PODCAST_FILE=$(find "$PROJECT_DIR/podcast-audio" -name "stt-finetune-podcast-*.mp3" -type f | head -n 1)
if [ -n "$PODCAST_FILE" ]; then
    echo "Full podcast: $PODCAST_FILE"
    FILE_SIZE=$(du -h "$PODCAST_FILE" | cut -f1)
    echo "Size: $FILE_SIZE"
fi
