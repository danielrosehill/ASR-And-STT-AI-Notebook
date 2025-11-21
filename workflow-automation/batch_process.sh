#!/bin/bash
# Batch processing script - process all prompts once and exit

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROCESSOR="$SCRIPT_DIR/prompt_processor.py"

echo "🔄 Batch Processing All Prompts"
echo "================================"
echo ""

python3 "$PROCESSOR" "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Batch processing complete!"
else
    echo ""
    echo "❌ Batch processing failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
