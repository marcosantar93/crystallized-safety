# Docker Build Guide

This guide explains how to build Docker images for the crystallized-safety experiments using the automated docker_ec2_builder tool.

## Overview

The project uses **docker_ec2_builder** to automate Docker image builds on EC2. This provides:

- **Smart instance selection** - Automatically chooses optimal EC2 instance based on Dockerfile complexity
- **Cost optimization** - Uses spot instances for up to 90% savings
- **Automated workflow** - Provisions EC2, builds, pushes to registry, and terminates automatically
- **Better reliability** - Proper SSH/SCP operations instead of user-data embedding

## Prerequisites

### 1. Environment Variables

Set your Docker Hub token:

```bash
export DOCKERHUB_TOKEN='your-dockerhub-token-here'
```

Get a token at: https://hub.docker.com/settings/security

**Note:** If you've set up centralized credentials in `~/.api_credentials`, this will be loaded automatically when you open a new terminal.

### 2. AWS Configuration

Ensure AWS CLI is configured:

```bash
aws configure
```

You need:
- AWS access key ID
- AWS secret access key
- Default region (e.g., us-east-1)
- EC2 key pair (e.g., Ec2tutorial)

### 3. Docker EC2 Builder Installation

The docker_ec2_builder should already be installed. If not:

```bash
cd ~/docker_ec2_builder
pip install -e .
```

## Building the Empathy Experiment Image

### Method 1: Shell Script (Recommended)

```bash
cd ~/paladin_claude/crystallized-safety
./build_docker_empathy.sh
```

This runs the Python build script automatically.

### Method 2: Python Script Directly

```bash
cd ~/paladin_claude/crystallized-safety
python3 build_docker_empathy.py
```

### What Happens During Build

1. **Analysis Phase** (< 1 second)
   - Analyzes Dockerfile.empathy from ~/runpod_experiments
   - Determines optimal EC2 instance type based on complexity
   - For empathy experiments: typically t3.medium or t3.large

2. **Provisioning Phase** (~1-2 minutes)
   - Launches spot EC2 instance (90% cheaper than on-demand)
   - Waits for instance to become ready
   - Sets up SSH connection

3. **Build Phase** (~5-10 minutes)
   - Uploads build context via SCP (Dockerfile + support files)
   - Runs docker build on EC2 instance
   - Much faster than building locally on Mac

4. **Push Phase** (~3-5 minutes)
   - Authenticates with Docker Hub
   - Pushes image to marcosantar93/crystallized-safety:empathy
   - Verifies push success

5. **Cleanup Phase** (< 1 minute)
   - Automatically terminates EC2 instance
   - No manual cleanup needed

**Total time:** ~10-18 minutes
**Total cost:** ~$0.02-0.05 (using spot instances)

## Build Output

### Success

```
✅ BUILD SUCCESSFUL
========================================

Image URI: marcosantar93/crystallized-safety:empathy

Next steps:
  1. Launch experiments: cd experiments && python launch_empathy_runpod.py
  2. Monitor results
  3. Analyze data when complete
```

### Failure

If the build fails, you'll see:
- Error message explaining what went wrong
- Instance ID (instance will auto-terminate)
- Suggestions for fixing the issue

## Comparing to Old Build Method

### Old Method (build_empathy_docker_ec2.sh)

```bash
cd ~/runpod_experiments
./build_empathy_docker_ec2.sh
```

**Issues:**
- Always uses t3.xlarge (potentially oversized/expensive)
- Embeds all files inline in user-data (brittle)
- No spot instance support
- Less flexible
- ~$0.06-0.10 per build

### New Method (build_docker_empathy.py)

```bash
cd ~/paladin_claude/crystallized-safety
./build_docker_empathy.sh
```

**Improvements:**
- ✅ Automatically selects right-sized instance
- ✅ Supports spot instances (90% savings)
- ✅ Uses proper SCP for file transfer
- ✅ Better error handling
- ✅ More reliable
- ✅ ~$0.02-0.05 per build

## Monitoring Builds

The build script will output the EC2 instance ID and provide a console link:

```
Instance ID: i-0123456789abcdef0
Console: https://console.aws.amazon.com/ec2/v2/home?region=us-east-1#Instances:instanceId=i-0123456789abcdef0
```

You can monitor the build in the AWS Console, but it's not necessary - the script will wait and report results automatically.

## Troubleshooting

### Error: DOCKERHUB_TOKEN not set

```bash
export DOCKERHUB_TOKEN='your-token'
# Or reload credentials
source ~/.zshrc
```

### Error: docker_ec2_builder not installed

```bash
cd ~/docker_ec2_builder
pip install -e .
```

### Error: Dockerfile not found

Make sure ~/runpod_experiments contains:
- Dockerfile.empathy
- empathy_prompts_v1.json
- empathy_experiment_main.py
- empathy_entrypoint.sh

### Build fails on EC2

Check the error message. Common issues:
- Docker Hub authentication failed (check token)
- Out of disk space (unlikely with docker_ec2_builder)
- Network issues (retry)

## Advanced Usage

### Build Other Dockerfiles

You can modify `build_docker_empathy.py` to build other Dockerfiles:

```python
DOCKERFILE_PATH = RUNPOD_EXPERIMENTS_DIR / "Dockerfile.cycles"
IMAGE_TAG = "marcosantar93/crystallized-safety:cycles"
```

### Use On-Demand Instances

Edit `build_docker_empathy.py`:

```python
USE_SPOT = False  # Use on-demand instead of spot
```

### Force Specific Instance Type

```python
builder = DockerEC2Builder(
    dockerfile_path=str(DOCKERFILE_PATH),
    image_tag=IMAGE_TAG,
    instance_type_override="c6i.xlarge",  # Force this type
    use_spot=USE_SPOT
)
```

## Next Steps After Build

Once the image is built and pushed:

1. **Launch Empathy Experiments:**
   ```bash
   cd experiments
   python launch_empathy_runpod.py
   ```

2. **Monitor pods** until complete

3. **Download and analyze results:**
   ```bash
   python analyze_empathy_results.py
   python generate_empathy_figures.py
   python create_empathy_report.py
   ```

## Related Documentation

- `~/docker_ec2_builder/README.md` - docker_ec2_builder documentation
- `EMPATHY_EXPERIMENT_STATUS.md` - Empathy experiment status
- `CONTINUE_HERE.md` - Session continuation guide

## Questions?

The docker_ec2_builder tool is located at:
```
~/docker_ec2_builder
```

Configuration file:
```
~/docker_ec2_builder/config.yaml
```
