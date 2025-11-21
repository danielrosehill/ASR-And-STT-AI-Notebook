#!/bin/bash
# Continuous monitoring script for prompt processing
# Uses inotifywait to detect new prompts and process them automatically

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TO_RUN_DIR="$SCRIPT_DIR/../prompts/to-run"
PROCESSOR="$SCRIPT_DIR/prompt_processor.py"

echo "🔍 Watching for new prompts in: $TO_RUN_DIR"
echo "📝 Press Ctrl+C to stop"
echo ""

# Check if inotify-tools is installed
if ! command -v inotifywait &> /dev/null; then
    echo "❌ Error: inotifywait not found"
    echo "Install with: sudo apt install inotify-tools"
    exit 1
fi

# Initial processing of any existing prompts
echo "🚀 Processing any existing prompts..."
python3 "$PROCESSOR"
echo ""

# Watch for new files
inotifywait -m -e close_write -e moved_to "$TO_RUN_DIR" --format '%w%f' |
while read filepath; do
    # Check if it's a text or markdown file
    if [[ "$filepath" =~ \.(txt|md)$ ]]; then
        filename=$(basename "$filepath")
        echo ""
        echo "🆕 New prompt detected: $filename"
        echo "⏳ Processing..."

        # Give file system a moment to settle
        sleep 1

        # Process the specific prompt
        python3 "$PROCESSOR" --prompt "$filename"

        echo ""
        echo "✅ Ready for next prompt..."
    fi
done
