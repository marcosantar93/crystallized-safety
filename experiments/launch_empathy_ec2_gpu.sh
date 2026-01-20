#!/bin/bash
# Launch EC2 GPU instance to run empathy experiments
set -e

echo "=========================================="
echo "EMPATHY EXPERIMENTS ON EC2 GPU"
echo "=========================================="
echo ""

REGION="${AWS_DEFAULT_REGION:-us-east-1}"
INSTANCE_TYPE="g4dn.xlarge"  # NVIDIA T4, $0.526/hr
IMAGE_NAME="marcosantar93/crystallized-safety:empathy"
KEY_NAME="Ec2tutorial"

if [ -z "$DOCKERHUB_TOKEN" ]; then
    echo "❌ ERROR: DOCKERHUB_TOKEN not set"
    exit 1
fi

echo "Configuration:"
echo "  Instance: $INSTANCE_TYPE (NVIDIA T4 GPU)"
echo "  Cost: ~$0.53/hour"
echo "  Image: $IMAGE_NAME"
echo ""

# Find Ubuntu 20.04 AMI
echo "Finding Ubuntu AMI with GPU support..."
AMI_ID=$(aws ec2 describe-images \
    --region $REGION \
    --owners 099720109477 \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*" "Name=state,Values=available" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text)

echo "  AMI: $AMI_ID"

# Security group
SG_ID=$(aws ec2 describe-security-groups \
    --region $REGION \
    --filters "Name=group-name,Values=docker-builder-sg" \
    --query 'SecurityGroups[0].GroupId' \
    --output text 2>/dev/null)

echo "  Security Group: $SG_ID"
echo ""

# Create user data that runs all experiments
cat > /tmp/empathy_gpu_userdata.sh << 'EOF'
#!/bin/bash
exec > >(tee /var/log/empathy-experiments.log)
exec 2>&1

echo "=========================================="
echo "EMPATHY EXPERIMENTS STARTING"
echo "=========================================="
date
echo ""

# Install NVIDIA drivers and Docker
echo "Installing NVIDIA drivers..."
apt-get update
apt-get install -y ubuntu-drivers-common
ubuntu-drivers autoinstall

echo "Installing Docker..."
apt-get install -y docker.io nvidia-docker2
systemctl restart docker

echo "Verifying GPU..."
nvidia-smi

# Login to DockerHub
echo "$DOCKERHUB_TOKEN" | docker login -u marcosantar93 --password-stdin

# Pull the image
echo "Pulling Docker image..."
docker pull marcosantar93/crystallized-safety:empathy

# Create results directory
mkdir -p /home/ubuntu/results

# Models to run
MODELS=("llama-3.1-8b" "qwen2.5-7b" "mistral-7b" "gemma2-9b" "deepseek-r1-7b")
MODEL_PATHS=(
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "google/gemma-2-9b-it"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-7B"
)

# Run each model sequentially
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_PATH="${MODEL_PATHS[$i]}"

    echo ""
    echo "=========================================="
    echo "RUNNING: $MODEL ($((i+1))/5)"
    echo "=========================================="
    date

    docker run --rm --gpus all \
        -e MODEL_NAME="$MODEL" \
        -e MODEL_PATH="$MODEL_PATH" \
        -e OUTPUT_DIR="/workspace/results" \
        -v /home/ubuntu/results:/workspace/results \
        marcosantar93/crystallized-safety:empathy

    if [ $? -eq 0 ]; then
        echo "✅ $MODEL completed"
    else
        echo "❌ $MODEL failed"
    fi
done

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=========================================="
date
echo ""

# Show results
echo "Results collected:"
ls -lh /home/ubuntu/results/*.json

# Compress results
cd /home/ubuntu
tar -czf empathy_results.tar.gz results/
echo ""
echo "Results archived: /home/ubuntu/empathy_results.tar.gz"
echo ""

# Auto-terminate
echo "Auto-terminating instance in 60 seconds..."
sleep 60

INSTANCE_ID=$(ec2-metadata --instance-id | cut -d' ' -f2)
REGION=$(ec2-metadata --availability-zone | cut -d' ' -f2 | sed 's/.$//')

aws ec2 terminate-instances --region $REGION --instance-ids $INSTANCE_ID
echo "✅ Termination initiated"
EOF

# Launch instance
echo "Launching GPU instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --region $REGION \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SG_ID \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --user-data file:///tmp/empathy_gpu_userdata.sh \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=empathy-experiments-gpu},{Key=AutoTerminate,Value=true}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

if [ -z "$INSTANCE_ID" ]; then
    echo "❌ Failed to launch instance"
    exit 1
fi

echo "✅ Instance launched: $INSTANCE_ID"
echo ""

# Wait for running
echo "Waiting for instance to start..."
aws ec2 wait instance-running --region $REGION --instance-ids $INSTANCE_ID

PUBLIC_IP=$(aws ec2 describe-instances \
    --region $REGION \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "✅ Instance running at: $PUBLIC_IP"
echo ""

echo "=========================================="
echo "EXPERIMENTS RUNNING"
echo "=========================================="
echo ""
echo "Instance Details:"
echo "  ID: $INSTANCE_ID"
echo "  IP: $PUBLIC_IP"
echo "  Type: $INSTANCE_TYPE (NVIDIA T4)"
echo "  Cost: ~$0.53/hour"
echo ""
echo "Expected timeline:"
echo "  - Setup: 2-3 minutes"
echo "  - Pull image: 1-2 minutes"
echo "  - Run 5 models: ~60-75 minutes (12-15 min each)"
echo "  - Auto-terminate: +60 seconds"
echo "  Total: ~70-80 minutes"
echo ""
echo "Monitor logs:"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP} 'tail -f /var/log/empathy-experiments.log'"
echo ""
echo "Download results when complete:"
echo "  scp -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}:/home/ubuntu/empathy_results.tar.gz ."
echo ""
echo "⚠️  Instance will AUTO-TERMINATE after completion"
echo ""
echo "Monitor in console:"
echo "  https://console.aws.amazon.com/ec2/v2/home?region=${REGION}#Instances:instanceId=${INSTANCE_ID}"
echo ""
echo "=========================================="
