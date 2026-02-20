#!/bin/bash
# Sync notebooks to Google Drive for Colab access
#
# Prerequisites:
#   brew install --cask google-drive
#   (Then sign in to Google Drive desktop app)
#
# Usage:
#   ./scripts/sync_to_drive.sh

DRIVE_PATH="$HOME/Library/CloudStorage/GoogleDrive-*/My Drive/paladin_claude"

# Expand the glob
DRIVE_DIR=$(echo $DRIVE_PATH)

if [[ ! -d "$DRIVE_DIR" ]]; then
    echo "Google Drive folder not found at: $DRIVE_PATH"
    echo ""
    echo "To set up Google Drive sync:"
    echo "  1. Install: brew install --cask google-drive"
    echo "  2. Sign in to Google Drive desktop app"
    echo "  3. Create folder: My Drive/paladin_claude"
    echo "  4. Run this script again"
    exit 1
fi

REPO_ROOT="$(dirname "$0")/.."
cd "$REPO_ROOT"

echo "Syncing to Google Drive..."
echo "  From: $(pwd)"
echo "  To:   $DRIVE_DIR"
echo ""

# Sync notebooks (excluding archives and checkpoints)
rsync -av --progress \
    --include='*.ipynb' \
    --include='*.py' \
    --include='*.json' \
    --include='*.md' \
    --include='*/' \
    --exclude='_archive/' \
    --exclude='_results_archive/' \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='.ipynb_checkpoints/' \
    . "$DRIVE_DIR/"

echo ""
echo "Sync complete!"
echo ""
echo "Your notebooks are now available in Colab at:"
echo "  https://colab.research.google.com/drive/"
echo ""
echo "Or open Google Drive > paladin_claude > [notebook].ipynb"
