#!/bin/bash
# Launch 5 EC2 GPU instances in PARALLEL - one per model
set -e

echo "=========================================="
echo "EMPATHY EXPERIMENTS - PARALLEL EC2"
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
echo "  Strategy: 5 instances in PARALLEL"
echo "  Instance: $INSTANCE_TYPE (NVIDIA T4)"
echo "  Cost: ~$2.65/hour (5 instances)"
echo "  Time: ~15-20 minutes (vs 70-80 sequential)"
echo ""

# Find AMI
AMI_ID=$(aws ec2 describe-images \
    --region $REGION \
    --owners 099720109477 \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*" "Name=state,Values=available" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text)

SG_ID=$(aws ec2 describe-security-groups \
    --region $REGION \
    --filters "Name=group-name,Values=docker-builder-sg" \
    --query 'SecurityGroups[0].GroupId' \
    --output text 2>/dev/null)

echo "  AMI: $AMI_ID"
echo "  Security Group: $SG_ID"
echo ""

# Model configurations (arrays)
MODEL_NAMES=("llama-3.1-8b" "qwen2.5-7b" "mistral-7b" "gemma2-9b" "deepseek-r1-7b")
MODEL_PATHS=(
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "google/gemma-2-9b-it"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-7B"
)

# Launch all instances in parallel
echo "Launching 5 instances in parallel..."
echo ""

INSTANCE_IDS=()
for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    MODEL_PATH="${MODEL_PATHS[$i]}"

    echo "🚀 Launching: $MODEL_NAME"

    # Create user data for this specific model
    cat > /tmp/userdata_${MODEL_NAME}.sh << EOF
#!/bin/bash
exec > >(tee /var/log/empathy-${MODEL_NAME}.log)
exec 2>&1

echo "=========================================="
echo "EMPATHY EXPERIMENT: ${MODEL_NAME}"
echo "=========================================="
date

# Install NVIDIA drivers
apt-get update
apt-get install -y ubuntu-drivers-common
ubuntu-drivers autoinstall

# Install Docker
apt-get install -y docker.io nvidia-docker2
systemctl restart docker

# Verify GPU
nvidia-smi

# Login to DockerHub
echo "$DOCKERHUB_TOKEN" | docker login -u marcosantar93 --password-stdin

# Pull image
docker pull ${IMAGE_NAME}

# Create results directory
mkdir -p /home/ubuntu/results

# Run experiment
echo ""
echo "Running experiment for ${MODEL_NAME}..."
docker run --rm --gpus all \
    -e MODEL_NAME="${MODEL_NAME}" \
    -e MODEL_PATH="${MODEL_PATH}" \
    -e OUTPUT_DIR="/workspace/results" \
    -v /home/ubuntu/results:/workspace/results \
    ${IMAGE_NAME}

EXIT_CODE=\$?

if [ \$EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ ${MODEL_NAME} COMPLETED"
    ls -lh /home/ubuntu/results/*.json
else
    echo "❌ ${MODEL_NAME} FAILED"
fi

# Archive results
cd /home/ubuntu
tar -czf empathy_${MODEL_NAME}_results.tar.gz results/

echo ""
echo "Results: /home/ubuntu/empathy_${MODEL_NAME}_results.tar.gz"

# Auto-terminate
echo "Auto-terminating in 60 seconds..."
sleep 60

INSTANCE_ID=\$(ec2-metadata --instance-id | cut -d' ' -f2)
REGION=\$(ec2-metadata --availability-zone | cut -d' ' -f2 | sed 's/.\$//')

aws ec2 terminate-instances --region \$REGION --instance-ids \$INSTANCE_ID
EOF

    # Launch instance
    INSTANCE_ID=$(aws ec2 run-instances \
        --region $REGION \
        --image-id $AMI_ID \
        --instance-type $INSTANCE_TYPE \
        --key-name $KEY_NAME \
        --security-group-ids $SG_ID \
        --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
        --user-data file:///tmp/userdata_${MODEL_NAME}.sh \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=empathy-${MODEL_NAME}},{Key=Model,Value=${MODEL_NAME}}]" \
        --query 'Instances[0].InstanceId' \
        --output text)

    INSTANCE_IDS+=("$INSTANCE_ID")
    echo "  ✅ $MODEL_NAME: $INSTANCE_ID"

    sleep 2  # Small delay between launches
done

echo ""
echo "=========================================="
echo "ALL INSTANCES LAUNCHED"
echo "=========================================="
echo ""

# Save instance info
cat > /tmp/empathy_parallel_instances.json << EOF
{
  "timestamp": "$(date -Iseconds)",
  "instances": {
EOF

for i in "${!MODEL_NAMES[@]}"; do
    if [ $i -gt 0 ]; then
        echo "," >> /tmp/empathy_parallel_instances.json
    fi
    echo "    \"${MODEL_NAMES[$i]}\": \"${INSTANCE_IDS[$i]}\"" >> /tmp/empathy_parallel_instances.json
done

cat >> /tmp/empathy_parallel_instances.json << EOF

  }
}
EOF

cp /tmp/empathy_parallel_instances.json ~/paladin_claude/crystallized-safety/experiments/results/empathy/

echo "Instance IDs:"
for i in "${!MODEL_NAMES[@]}"; do
    echo "  ${MODEL_NAMES[$i]}: ${INSTANCE_IDS[$i]}"
done

echo ""
echo "Expected timeline:"
echo "  - Setup & drivers: 5-7 minutes"
echo "  - Pull image: 1-2 minutes"
echo "  - Run experiment: 8-12 minutes"
echo "  - Total per model: ~15-20 minutes"
echo ""
echo "All running in parallel - results in ~20 minutes!"
echo ""
echo "Cost: ~\$0.50 total (5 instances × ~20 min × \$0.53/hr)"
echo ""
echo "Monitor: Check AWS Console or wait for auto-termination"
echo "  https://console.aws.amazon.com/ec2/v2/home?region=${REGION}#Instances:"
echo ""
echo "=========================================="
