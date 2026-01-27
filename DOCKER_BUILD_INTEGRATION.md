# Docker EC2 Builder Integration

**Date:** 2026-01-21
**Status:** ✅ Complete and ready to use

## What Changed

Integrated the `~/docker_ec2_builder` tool into the crystallized-safety project to provide automated, optimized Docker image builds on EC2.

## New Files Created

### 1. `build_docker_empathy.py`
Python script that uses docker_ec2_builder to build the empathy experiment Docker image.

**Features:**
- Automatic instance type selection based on Dockerfile analysis
- Spot instance support for 90% cost savings
- Proper SSH/SCP file transfer (vs user-data embedding)
- Auto-terminating instances
- Better error handling

**Usage:**
```bash
cd ~/paladin_claude/crystallized-safety
python3 build_docker_empathy.py
```

### 2. `build_docker_empathy.sh`
Convenient shell wrapper for the Python script.

**Usage:**
```bash
cd ~/paladin_claude/crystallized-safety
./build_docker_empathy.sh
```

### 3. `BUILD_GUIDE.md`
Comprehensive documentation covering:
- Prerequisites and setup
- Build process walkthrough
- Comparison with old method
- Troubleshooting guide
- Advanced usage examples

## Environment Variables Required

You need to set **`DOCKERHUB_TOKEN`** (already configured in your `~/.api_credentials`):

```bash
export DOCKERHUB_TOKEN='your-dockerhub-token-here'
```

Get a token at: https://hub.docker.com/settings/security

Since you've set up centralized credentials, this will be loaded automatically when you open a new terminal. Just reload:

```bash
source ~/.zshrc
```

## Comparison: Old vs New

### Old Method
```bash
cd ~/runpod_experiments
./build_empathy_docker_ec2.sh
```

**Limitations:**
- Always uses t3.xlarge (fixed size, potentially wasteful)
- Embeds files inline in user-data (error-prone)
- No spot instance support
- Cost: ~$0.06-0.10 per build

### New Method
```bash
cd ~/paladin_claude/crystallized-safety
./build_docker_empathy.sh
```

**Improvements:**
- ✅ Smart instance selection (right-sized for workload)
- ✅ Spot instances (90% savings)
- ✅ Proper SCP file transfer
- ✅ Better error handling
- ✅ More reliable
- ✅ Cost: ~$0.02-0.05 per build

## How It Works

1. **Analysis** - Analyzes Dockerfile complexity
   - Base image weight
   - Number of RUN commands
   - Package installations
   - Compilation requirements
   - GPU requirements

2. **Provisioning** - Launches optimal EC2 instance
   - Uses spot instances by default
   - Auto-selects instance type
   - For empathy: typically t3.medium or t3.large

3. **Building** - Builds Docker image on EC2
   - Uploads build context via SCP
   - Runs docker build
   - Much faster than local Mac builds

4. **Pushing** - Pushes to Docker Hub
   - Authenticates with DOCKERHUB_TOKEN
   - Pushes to marcosantar93/crystallized-safety:empathy
   - Verifies success

5. **Cleanup** - Auto-terminates instance
   - No manual cleanup needed
   - Prevents runaway costs

**Total time:** ~10-18 minutes
**Total cost:** ~$0.02-0.05

## Quick Start

### Prerequisites Check

```bash
# 1. Check Docker Hub token is set
echo $DOCKERHUB_TOKEN

# If empty, reload credentials
source ~/.zshrc

# 2. Check AWS is configured
aws sts get-caller-identity

# 3. Check docker_ec2_builder is installed
python3 -c "import docker_ec2_builder; print('✅ Installed')"

# If not installed:
cd ~/docker_ec2_builder
pip install -e .
```

### Build Empathy Docker Image

```bash
cd ~/paladin_claude/crystallized-safety
./build_docker_empathy.sh
```

That's it! The script will:
- Verify prerequisites
- Analyze Dockerfile
- Provision spot EC2 instance
- Build Docker image
- Push to Docker Hub
- Terminate instance
- Print next steps

## Next Steps After Build

Once the image is built:

1. **Launch experiments:**
   ```bash
   cd experiments
   python launch_empathy_runpod.py
   ```

2. **Monitor pods** until complete

3. **Analyze results:**
   ```bash
   python analyze_empathy_results.py
   python generate_empathy_figures.py
   python create_empathy_report.py
   ```

## Integration with Existing Project

The new build scripts integrate seamlessly:

- **Source files:** Still in `~/runpod_experiments`
  - Dockerfile.empathy
  - empathy_prompts_v1.json
  - empathy_experiment_main.py
  - empathy_entrypoint.sh

- **Build scripts:** Now in `~/paladin_claude/crystallized-safety`
  - build_docker_empathy.py (new)
  - build_docker_empathy.sh (new)
  - BUILD_GUIDE.md (new)

- **Old scripts:** Still available if needed
  - ~/runpod_experiments/build_empathy_docker_ec2.sh

You can use either method, but the new one is recommended for:
- Cost savings (spot instances)
- Reliability (proper SCP transfer)
- Flexibility (automatic instance sizing)

## Docker EC2 Builder Configuration

The docker_ec2_builder is configured at:
```
~/docker_ec2_builder/config.yaml
```

Current settings:
- Region: us-east-1
- Key pair: Ec2tutorial
- Security group: sg-0b6bbcac4567fcd67
- Spot instances: enabled
- Registry: ECR (but overridden to Docker Hub in scripts)

You generally don't need to modify this - the Python script handles configuration.

## Troubleshooting

See `BUILD_GUIDE.md` for comprehensive troubleshooting, or:

### Common Issues

1. **DOCKERHUB_TOKEN not set**
   ```bash
   source ~/.zshrc
   echo $DOCKERHUB_TOKEN  # Should print your token
   ```

2. **docker_ec2_builder not found**
   ```bash
   cd ~/docker_ec2_builder
   pip install -e .
   ```

3. **AWS credentials not configured**
   ```bash
   aws configure
   ```

## Documentation

- **BUILD_GUIDE.md** - Comprehensive build guide (this project)
- **~/docker_ec2_builder/README.md** - docker_ec2_builder documentation
- **~/docker_ec2_builder/claude.md** - Detailed docker_ec2_builder docs
- **EMPATHY_EXPERIMENT_STATUS.md** - Empathy experiment status

## Ready to Use! 🚀

Everything is set up and ready. When you want to build the empathy Docker image:

```bash
cd ~/paladin_claude/crystallized-safety
./build_docker_empathy.sh
```

The old method still works if needed, but the new method is recommended for cost and reliability.
