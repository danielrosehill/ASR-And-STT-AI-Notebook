#!/bin/bash
# Install systemd service for automatic prompt processing

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_FILE="$SCRIPT_DIR/stt-prompt-processor.service"
SYSTEMD_DIR="/etc/systemd/system"

echo "🔧 STT Prompt Processor Service Installer"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "❌ Don't run this script as root/sudo"
    echo "The script will prompt for sudo when needed"
    exit 1
fi

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠️  ANTHROPIC_API_KEY not set in environment"
    echo ""
    read -p "Enter your Anthropic API key: " API_KEY

    if [ -z "$API_KEY" ]; then
        echo "❌ API key cannot be empty"
        exit 1
    fi

    # Update service file with API key
    sed -i "s/YOUR_API_KEY_HERE/$API_KEY/" "$SERVICE_FILE"
else
    echo "✅ Using ANTHROPIC_API_KEY from environment"
    sed -i "s/YOUR_API_KEY_HERE/$ANTHROPIC_API_KEY/" "$SERVICE_FILE"
fi

# Copy service file
echo ""
echo "📋 Installing service file..."
sudo cp "$SERVICE_FILE" "$SYSTEMD_DIR/"

if [ $? -ne 0 ]; then
    echo "❌ Failed to copy service file"
    exit 1
fi

# Reload systemd
echo "🔄 Reloading systemd daemon..."
sudo systemctl daemon-reload

# Enable service
echo "⚙️  Enabling service..."
sudo systemctl enable stt-prompt-processor.service

# Start service
echo "🚀 Starting service..."
sudo systemctl start stt-prompt-processor.service

# Check status
echo ""
echo "📊 Service Status:"
sudo systemctl status stt-prompt-processor.service --no-pager

echo ""
echo "✅ Installation complete!"
echo ""
echo "Service commands:"
echo "  sudo systemctl start stt-prompt-processor    # Start service"
echo "  sudo systemctl stop stt-prompt-processor     # Stop service"
echo "  sudo systemctl restart stt-prompt-processor  # Restart service"
echo "  sudo systemctl status stt-prompt-processor   # Check status"
echo "  sudo journalctl -u stt-prompt-processor -f   # View logs"
echo ""
echo "The service will automatically start on boot and watch for new prompts."
