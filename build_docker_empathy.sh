#!/bin/bash
# Wrapper script for building empathy Docker image
# Uses docker_ec2_builder for automated, optimized EC2 builds

set -e

echo "=========================================="
echo "EMPATHY DOCKER BUILDER"
echo "=========================================="
echo ""

# Check if running from correct directory
if [ ! -f "build_docker_empathy.py" ]; then
    echo "❌ Error: Must run from crystallized-safety directory"
    exit 1
fi

# Check DOCKERHUB_TOKEN
if [ -z "$DOCKERHUB_TOKEN" ]; then
    echo "❌ Error: DOCKERHUB_TOKEN not set"
    echo ""
    echo "Set it with:"
    echo "  export DOCKERHUB_TOKEN='your-token'"
    echo ""
    echo "Or reload your shell configuration:"
    echo "  source ~/.zshrc"
    exit 1
fi

# Run Python build script
python3 build_docker_empathy.py
