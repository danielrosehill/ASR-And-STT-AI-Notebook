#!/bin/bash
# Helper script to create a new prompt with proper naming

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TO_RUN_DIR="$SCRIPT_DIR/../prompts/to-run"

# Get current date for naming
DATE=$(date +%Y%m%d)

echo "📝 Create New Prompt"
echo "===================="
echo ""

# Get topic keywords from user
read -p "Enter topic keywords (e.g., 'model-comparison'): " TOPIC

if [ -z "$TOPIC" ]; then
    echo "❌ Error: Topic cannot be empty"
    exit 1
fi

# Clean up topic (lowercase, replace spaces with hyphens)
TOPIC_CLEAN=$(echo "$TOPIC" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | sed 's/[^a-z0-9-]//g')

# Choose extension
echo ""
echo "Select file format:"
echo "1) .txt (plain text)"
echo "2) .md (markdown)"
read -p "Choice [1/2]: " FORMAT_CHOICE

case $FORMAT_CHOICE in
    1)
        EXT="txt"
        ;;
    2)
        EXT="md"
        ;;
    *)
        EXT="txt"
        ;;
esac

# Generate filename
FILENAME="${DATE}-${TOPIC_CLEAN}.${EXT}"
FILEPATH="$TO_RUN_DIR/$FILENAME"

# Check if file exists
if [ -f "$FILEPATH" ]; then
    echo ""
    echo "⚠️  File already exists: $FILENAME"
    read -p "Overwrite? [y/N]: " OVERWRITE
    if [[ ! "$OVERWRITE" =~ ^[Yy]$ ]]; then
        echo "Cancelled"
        exit 0
    fi
fi

# Get prompt content
echo ""
echo "Enter your prompt/question (Ctrl+D when done):"
echo "----------------------------------------"

# Read multiline input
CONTENT=$(cat)

# Write to file
echo "$CONTENT" > "$FILEPATH"

echo ""
echo "✅ Prompt created: $FILEPATH"
echo ""
echo "The prompt will be automatically processed if watch mode is running,"
echo "or you can manually process with:"
echo "  ./batch_process.sh --prompt $FILENAME"
