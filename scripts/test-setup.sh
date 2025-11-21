#!/bin/bash
# Test script to verify podcast generation setup

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "Podcast Generation Setup Test"
echo "========================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
}

check_warn() {
    echo -e "${YELLOW}!${NC} $1"
}

ERRORS=0
WARNINGS=0

# Check Python
echo "Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    check_pass "Python 3 found: $PYTHON_VERSION"
else
    check_fail "Python 3 not found"
    ((ERRORS++))
fi

# Check OpenRouter API key
echo ""
echo "Checking environment variables..."
if [ -n "$OPENROUTER_API_KEY" ]; then
    check_pass "OPENROUTER_API_KEY is set"
else
    check_warn "OPENROUTER_API_KEY not set (required for Stage 1)"
    echo "  Set with: export OPENROUTER_API_KEY='your-key-here'"
    ((WARNINGS++))
fi

# Check Python dependencies
echo ""
echo "Checking Python dependencies..."

check_python_package() {
    if python3 -c "import $1" 2>/dev/null; then
        check_pass "$1 installed"
    else
        check_warn "$1 not installed"
        echo "  Install with: pip install $1"
        ((WARNINGS++))
        return 1
    fi
}

check_python_package "requests"
check_python_package "edge_tts"

# Check ffmpeg
echo ""
echo "Checking system dependencies..."
if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version | head -n1 | cut -d' ' -f3)
    check_pass "ffmpeg found: $FFMPEG_VERSION"
else
    check_warn "ffmpeg not found (optional, for concatenation)"
    echo "  Install with: sudo apt install ffmpeg"
    ((WARNINGS++))
fi

# Check directory structure
echo ""
echo "Checking directory structure..."
if [ -d "$SCRIPT_DIR/../notebook" ]; then
    MD_COUNT=$(find "$SCRIPT_DIR/../notebook" -name "*.md" | wc -l)
    check_pass "notebook/ directory found with $MD_COUNT markdown files"
else
    check_fail "notebook/ directory not found"
    ((ERRORS++))
fi

# Check scripts
echo ""
echo "Checking scripts..."
for script in convert-to-ssml.py generate-podcast.py create-podcast.sh; do
    if [ -f "$SCRIPT_DIR/$script" ]; then
        if [ -x "$SCRIPT_DIR/$script" ]; then
            check_pass "$script found and executable"
        else
            check_warn "$script found but not executable"
            echo "  Fix with: chmod +x $SCRIPT_DIR/$script"
            ((WARNINGS++))
        fi
    else
        check_fail "$script not found"
        ((ERRORS++))
    fi
done

# Summary
echo ""
echo "========================================"
echo "Summary"
echo "========================================"
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "You're ready to generate podcasts!"
    echo "Run: ./create-podcast.sh"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}! Setup mostly complete with $WARNINGS warning(s)${NC}"
    echo ""
    echo "You can still proceed, but some features may not work."
    echo ""
    echo "To install missing Python packages:"
    echo "  pip install -r requirements.txt"
    echo ""
    echo "To proceed anyway:"
    echo "  ./create-podcast.sh"
    exit 0
else
    echo -e "${RED}✗ Setup incomplete with $ERRORS error(s) and $WARNINGS warning(s)${NC}"
    echo ""
    echo "Please fix the errors above before proceeding."
    exit 1
fi
