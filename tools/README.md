# EC2 Optimizer

Automatically analyze Docker images and spawn optimal EC2 instances.

## Features

- 📊 Analyzes Docker images to estimate resource requirements
- 💰 Recommends cheapest EC2 instances that meet requirements
- 🚀 Spawns instances with Docker image pre-configured
- 🎯 Supports CPU and GPU workloads
- 🔍 Dry-run mode to preview before launching

## Installation

```bash
pip install boto3 docker
```

## Quick Start

### 1. Analyze an image and get recommendations

```bash
python ec2_optimizer.py --image pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
```

Output:
```
TOP 5 RECOMMENDATIONS (Sorted by price)
1. g4dn.xlarge: 4 vCPUs, 16GB RAM, 1 GPUs (16GB), $0.53/hr
   Monthly cost (730 hrs): $383.35
   Daily cost (24 hrs): $12.62
```

### 2. Launch the best option

```bash
python ec2_optimizer.py \
  --image pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime \
  --spawn \
  --key-name my-key \
  --security-group sg-xxxxx
```

### 3. Dry run (preview without launching)

```bash
python ec2_optimizer.py \
  --image nvidia/cuda:12.0-base \
  --dry-run
```

## Usage Examples

### CPU-only workload

```bash
python ec2_optimizer.py --image python:3.9-slim
```

### GPU workload with custom requirements

```bash
python ec2_optimizer.py \
  --image custom/ml-experiment:latest \
  --memory 32 \
  --gpu required \
  --spawn
```

### Large language model (auto-detects 80GB GPU need)

```bash
python ec2_optimizer.py --image my-llm:70b
```

## Command Line Options

```
--image IMAGE          Docker image name (required)
--spawn                Actually spawn the instance (default: just recommend)
--dry-run              Show what would be launched without launching
--memory GB            Override minimum memory (GB)
--vcpus N              Override minimum vCPUs
--gpu OPTION           GPU requirement: required/optional/none
--region REGION        AWS region (default: us-east-1)
--key-name KEY         SSH key pair name
--security-group SG    Security group ID
--top-n N              Show top N recommendations (default: 5)
```

## How It Works

### 1. Image Analysis

The tool analyzes your Docker image to determine:

- **Image size** → Storage and memory needs
- **GPU requirements** → Detects CUDA, PyTorch, TensorFlow
- **Framework detection** → Adjusts vCPU/memory for ML workloads
- **Model size hints** → Detects 7B, 13B, 70B in names → GPU memory

### 2. Instance Selection

Matches requirements against EC2 catalog:

- **CPU instances**: t3, c5, r5 families
- **GPU instances**:
  - g4dn (T4, 16GB) - Cost-effective for inference
  - p3 (V100, 16GB) - Training workloads
  - p4d (A100, 40GB) - Large models

### 3. Instance Spawning

When you use `--spawn`:

1. Selects cheapest matching instance
2. Finds latest Amazon Linux 2 + Docker AMI
3. Launches instance with user data script
4. Auto-pulls Docker image on boot
5. Returns instance ID and public IP

## Cost Estimation

The tool shows:
- **Hourly rate**: On-demand pricing
- **Daily cost**: 24-hour runtime
- **Monthly cost**: 730-hour runtime

Example:
```
g4dn.xlarge: 4 vCPUs, 16GB RAM, 1 GPUs (16GB), $0.53/hr
   Monthly cost (730 hrs): $383.35
   Daily cost (24 hrs): $12.62
```

## Instance Catalog

### CPU Instances

| Type | vCPUs | RAM | Price/hr |
|------|-------|-----|----------|
| t3.micro | 2 | 1GB | $0.01 |
| t3.small | 2 | 2GB | $0.02 |
| t3.large | 2 | 8GB | $0.08 |
| c5.2xlarge | 8 | 16GB | $0.34 |
| r5.2xlarge | 8 | 64GB | $0.50 |

### GPU Instances

| Type | vCPUs | RAM | GPUs | GPU Mem | Price/hr |
|------|-------|-----|------|---------|----------|
| g4dn.xlarge | 4 | 16GB | 1 × T4 | 16GB | $0.53 |
| g4dn.12xlarge | 48 | 192GB | 4 × T4 | 64GB | $3.91 |
| p3.2xlarge | 8 | 61GB | 1 × V100 | 16GB | $3.06 |
| p4d.24xlarge | 96 | 1152GB | 8 × A100 | 320GB | $32.77 |

## Detection Logic

The tool auto-detects requirements from:

- **Image name**: `pytorch`, `cuda`, `7b`, `70b`
- **Environment variables**: `CUDA_VERSION`, `PYTORCH_VERSION`
- **Image labels**: Docker metadata
- **Image size**: Large images → More memory

### Examples:

```python
# Detects: GPU required, 16GB GPU mem
pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Detects: GPU required, 24GB GPU mem
custom/llm-inference:7b

# Detects: GPU required, 80GB GPU mem
llama-70b:latest

# Detects: CPU only, 8GB RAM
python:3.9-slim
```

## Security Best Practices

1. **Use Key Pairs**: Always specify `--key-name` for SSH access
2. **Security Groups**: Limit inbound access with `--security-group`
3. **IAM Roles**: Attach IAM roles for AWS resource access
4. **Auto-Shutdown**: Set up CloudWatch alarms for cost control

## Integration with Experiments

Use with the empathy geometry experiment:

```bash
# Spawn GPU instance for experiment
python tools/ec2_optimizer.py \
  --image pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime \
  --memory 32 \
  --spawn \
  --key-name experiment-key

# SSH into instance
ssh -i ~/.ssh/experiment-key.pem ec2-user@<PUBLIC_IP>

# Run experiment
docker run -it --gpus all \
  -v $(pwd):/workspace \
  pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime \
  python /workspace/experiments/empathy_experiment_main.py
```

## Troubleshooting

### "No matching instances found"

- Try `--memory 8` to lower memory requirement
- Use `--gpu optional` instead of `required`
- Check `--top-n 10` for more options

### "Docker not available"

- Tool will still work but use name-based heuristics
- Install Docker for full image analysis

### "AMI not found"

- Check region with `--region us-west-2`
- Ensure AWS credentials are configured

## Roadmap

Future enhancements:

- [ ] Spot instance support (70% cheaper)
- [ ] Auto-shutdown after N hours
- [ ] CloudFormation template generation
- [ ] Multi-region price comparison
- [ ] SageMaker integration
- [ ] Instance resize recommendations
- [ ] Cost alerts and budgets

## License

MIT License - Part of the Crystallized Safety Project
