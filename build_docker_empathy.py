#!/usr/bin/env python3
"""
Build empathy experiment Docker image using docker_ec2_builder.

This script simplifies Docker builds by:
- Automatically selecting optimal EC2 instance type
- Supporting spot instances for 90% cost savings
- Handling all SSH/SCP operations automatically
- Auto-terminating instances after build
"""

import os
import sys
from pathlib import Path

# Check if docker_ec2_builder is installed
try:
    from docker_ec2_builder import DockerEC2Builder
except ImportError:
    print("❌ Error: docker_ec2_builder not installed")
    print("\nInstall it with:")
    print("  cd ~/docker_ec2_builder")
    print("  pip install -e .")
    sys.exit(1)

# Configuration
RUNPOD_EXPERIMENTS_DIR = Path.home() / "runpod_experiments"
DOCKERFILE_PATH = RUNPOD_EXPERIMENTS_DIR / "Dockerfile.empathy"
IMAGE_TAG = "marcosantar93/crystallized-safety:empathy"
AWS_REGION = "us-east-1"
USE_SPOT = True  # Use spot instances for cost savings

def check_prerequisites():
    """Check that all prerequisites are met."""
    errors = []

    # Check DOCKERHUB_TOKEN
    if not os.environ.get("DOCKERHUB_TOKEN"):
        errors.append("DOCKERHUB_TOKEN environment variable not set")
        print("  Set it with: export DOCKERHUB_TOKEN='your-token'")
        print("  Get token at: https://hub.docker.com/settings/security")

    # Check Dockerfile exists
    if not DOCKERFILE_PATH.exists():
        errors.append(f"Dockerfile not found at {DOCKERFILE_PATH}")

    # Check required files exist
    required_files = [
        "empathy_prompts_v1.json",
        "empathy_experiment_main.py",
        "empathy_entrypoint.sh"
    ]

    for filename in required_files:
        filepath = RUNPOD_EXPERIMENTS_DIR / filename
        if not filepath.exists():
            errors.append(f"Required file not found: {filepath}")

    if errors:
        print("❌ Prerequisites check failed:\n")
        for error in errors:
            print(f"  - {error}")
        print()
        return False

    print("✅ All prerequisites met")
    return True

def main():
    print("=" * 60)
    print("EMPATHY DOCKER IMAGE BUILDER")
    print("=" * 60)
    print()

    # Check prerequisites
    print("Checking prerequisites...")
    if not check_prerequisites():
        sys.exit(1)
    print()

    # Configuration summary
    print("Configuration:")
    print(f"  Dockerfile: {DOCKERFILE_PATH}")
    print(f"  Image tag: {IMAGE_TAG}")
    print(f"  AWS region: {AWS_REGION}")
    print(f"  Use spot instances: {USE_SPOT}")
    print(f"  Registry: Docker Hub (marcosantar93)")
    print()

    # Initialize builder
    print("Initializing Docker EC2 Builder...")
    builder = DockerEC2Builder(
        dockerfile_path=str(DOCKERFILE_PATH),
        image_tag=IMAGE_TAG,
        aws_region=AWS_REGION,
        registry_type="dockerhub",
        use_spot=USE_SPOT
    )

    # Build the image
    print()
    print("Starting build process...")
    print("(This will analyze Dockerfile, provision EC2, build, push, and terminate)")
    print()

    result = builder.build()

    # Print results
    print()
    print("=" * 60)
    if result.success:
        print("✅ BUILD SUCCESSFUL")
        print("=" * 60)
        print()
        print(f"Image URI: {result.image_uri}")
        print()
        print("Next steps:")
        print("  1. Launch experiments: cd experiments && python launch_empathy_runpod.py")
        print("  2. Monitor results")
        print("  3. Analyze data when complete")
        print()
    else:
        print("❌ BUILD FAILED")
        print("=" * 60)
        print()
        print(f"Error: {result.error}")
        print()
        if result.instance_id:
            print(f"Instance ID: {result.instance_id}")
            print("(Instance should auto-terminate)")
        print()
        sys.exit(1)

if __name__ == "__main__":
    main()
