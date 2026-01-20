#!/usr/bin/env python3
"""
EC2 Optimizer: Estimate optimal EC2 instance type for a Docker image and spawn it

Usage:
    python ec2_optimizer.py --image pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime --spawn
    python ec2_optimizer.py --image custom/ml-experiment:latest --dry-run
    python ec2_optimizer.py --image nvidia/cuda:12.0-base --memory 32 --gpu required
"""

import argparse
import boto3
import docker
import json
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import subprocess
import time

@dataclass
class InstanceRequirements:
    """Resource requirements derived from Docker image analysis"""
    min_vcpus: int = 2
    min_memory_gb: int = 4
    min_gpu_memory_gb: int = 0
    gpu_required: bool = False
    architecture: str = "x86_64"  # or "arm64"
    storage_gb: int = 30

    def __repr__(self):
        return (f"InstanceRequirements(vcpus={self.min_vcpus}, "
                f"memory={self.min_memory_gb}GB, "
                f"gpu_mem={self.min_gpu_memory_gb}GB, "
                f"gpu_required={self.gpu_required}, "
                f"arch={self.architecture}, "
                f"storage={self.storage_gb}GB)")


@dataclass
class EC2InstanceOption:
    """EC2 instance type with pricing and specs"""
    instance_type: str
    vcpus: int
    memory_gb: float
    gpu_count: int
    gpu_memory_gb: int
    price_per_hour: float
    architecture: str

    def monthly_cost(self, hours_per_month: int = 730) -> float:
        """Estimate monthly cost"""
        return self.price_per_hour * hours_per_month

    def __repr__(self):
        return (f"{self.instance_type}: {self.vcpus} vCPUs, {self.memory_gb}GB RAM, "
                f"{self.gpu_count} GPUs ({self.gpu_memory_gb}GB), ${self.price_per_hour:.2f}/hr")


class DockerImageAnalyzer:
    """Analyze Docker images to estimate resource requirements"""

    def __init__(self):
        try:
            self.client = docker.from_env()
        except Exception as e:
            print(f"Warning: Docker not available: {e}")
            self.client = None

    def analyze_image(self, image_name: str) -> InstanceRequirements:
        """Analyze Docker image and estimate requirements"""
        req = InstanceRequirements()

        # Check if image exists locally or pull it
        try:
            if self.client:
                try:
                    image = self.client.images.get(image_name)
                except docker.errors.ImageNotFound:
                    print(f"Pulling image: {image_name}...")
                    image = self.client.images.pull(image_name)

                # Get image size
                image_size_mb = image.attrs['Size'] / (1024 * 1024)
                req.storage_gb = max(30, int(image_size_mb / 1024 * 3))  # 3x image size

                # Check image labels and environment variables
                config = image.attrs.get('Config', {})
                env_vars = config.get('Env', [])
                labels = config.get('Labels', {})

                # Detect GPU requirements
                image_lower = image_name.lower()
                env_str = ' '.join(env_vars).lower()
                labels_str = ' '.join(f"{k}={v}" for k, v in labels.items()).lower()

                gpu_keywords = ['cuda', 'cudnn', 'gpu', 'nvidia', 'torch+cu', 'tensorflow-gpu']
                if any(kw in image_lower or kw in env_str or kw in labels_str for kw in gpu_keywords):
                    req.gpu_required = True

                    # Estimate GPU memory based on image name
                    if 'pytorch' in image_lower or 'tensorflow' in image_lower:
                        req.min_gpu_memory_gb = 16  # Default for ML frameworks
                    if 'llm' in image_lower or '70b' in image_lower:
                        req.min_gpu_memory_gb = 80
                    elif '13b' in image_lower or '7b' in image_lower:
                        req.min_gpu_memory_gb = 24

                # Estimate CPU and memory based on image layers
                if image_size_mb > 5000:  # Large image (>5GB)
                    req.min_vcpus = 8
                    req.min_memory_gb = 32
                elif image_size_mb > 2000:  # Medium image (>2GB)
                    req.min_vcpus = 4
                    req.min_memory_gb = 16
                else:  # Small image
                    req.min_vcpus = 2
                    req.min_memory_gb = 8

                # Check for specific frameworks
                if 'pytorch' in image_lower or 'tensorflow' in image_lower:
                    req.min_memory_gb = max(req.min_memory_gb, 16)
                    req.min_vcpus = max(req.min_vcpus, 4)

                print(f"✓ Analyzed image: {image_name}")
                print(f"  Size: {image_size_mb:.1f} MB")
                print(f"  GPU required: {req.gpu_required}")

        except Exception as e:
            print(f"Warning: Could not analyze image: {e}")
            print(f"Using defaults based on image name...")

            # Fallback: analyze image name
            image_lower = image_name.lower()
            if any(kw in image_lower for kw in ['cuda', 'gpu', 'nvidia']):
                req.gpu_required = True
                req.min_gpu_memory_gb = 16

            if 'pytorch' in image_lower or 'tensorflow' in image_lower:
                req.min_vcpus = 4
                req.min_memory_gb = 16

        return req


class EC2InstanceSelector:
    """Select optimal EC2 instance type based on requirements"""

    # US-East-1 pricing (approximate, as of 2024)
    INSTANCE_CATALOG = {
        # CPU instances
        't3.micro': EC2InstanceOption('t3.micro', 2, 1, 0, 0, 0.0104, 'x86_64'),
        't3.small': EC2InstanceOption('t3.small', 2, 2, 0, 0, 0.0208, 'x86_64'),
        't3.medium': EC2InstanceOption('t3.medium', 2, 4, 0, 0, 0.0416, 'x86_64'),
        't3.large': EC2InstanceOption('t3.large', 2, 8, 0, 0, 0.0832, 'x86_64'),
        't3.xlarge': EC2InstanceOption('t3.xlarge', 4, 16, 0, 0, 0.1664, 'x86_64'),
        't3.2xlarge': EC2InstanceOption('t3.2xlarge', 8, 32, 0, 0, 0.3328, 'x86_64'),

        # Compute optimized
        'c5.large': EC2InstanceOption('c5.large', 2, 4, 0, 0, 0.085, 'x86_64'),
        'c5.xlarge': EC2InstanceOption('c5.xlarge', 4, 8, 0, 0, 0.17, 'x86_64'),
        'c5.2xlarge': EC2InstanceOption('c5.2xlarge', 8, 16, 0, 0, 0.34, 'x86_64'),
        'c5.4xlarge': EC2InstanceOption('c5.4xlarge', 16, 32, 0, 0, 0.68, 'x86_64'),

        # Memory optimized
        'r5.large': EC2InstanceOption('r5.large', 2, 16, 0, 0, 0.126, 'x86_64'),
        'r5.xlarge': EC2InstanceOption('r5.xlarge', 4, 32, 0, 0, 0.252, 'x86_64'),
        'r5.2xlarge': EC2InstanceOption('r5.2xlarge', 8, 64, 0, 0, 0.504, 'x86_64'),

        # GPU instances (P3 - V100)
        'p3.2xlarge': EC2InstanceOption('p3.2xlarge', 8, 61, 1, 16, 3.06, 'x86_64'),
        'p3.8xlarge': EC2InstanceOption('p3.8xlarge', 32, 244, 4, 64, 12.24, 'x86_64'),
        'p3.16xlarge': EC2InstanceOption('p3.16xlarge', 64, 488, 8, 128, 24.48, 'x86_64'),

        # GPU instances (G4 - T4)
        'g4dn.xlarge': EC2InstanceOption('g4dn.xlarge', 4, 16, 1, 16, 0.526, 'x86_64'),
        'g4dn.2xlarge': EC2InstanceOption('g4dn.2xlarge', 8, 32, 1, 16, 0.752, 'x86_64'),
        'g4dn.4xlarge': EC2InstanceOption('g4dn.4xlarge', 16, 64, 1, 16, 1.204, 'x86_64'),
        'g4dn.8xlarge': EC2InstanceOption('g4dn.8xlarge', 32, 128, 1, 16, 2.176, 'x86_64'),
        'g4dn.12xlarge': EC2InstanceOption('g4dn.12xlarge', 48, 192, 4, 64, 3.912, 'x86_64'),

        # GPU instances (P4 - A100)
        'p4d.24xlarge': EC2InstanceOption('p4d.24xlarge', 96, 1152, 8, 320, 32.77, 'x86_64'),
    }

    def select_instance(self, req: InstanceRequirements) -> List[EC2InstanceOption]:
        """Select best instances matching requirements"""
        matches = []

        for inst in self.INSTANCE_CATALOG.values():
            # Filter by architecture
            if inst.architecture != req.architecture:
                continue

            # Filter by GPU requirement
            if req.gpu_required:
                if inst.gpu_count == 0:
                    continue
                if inst.gpu_memory_gb < req.min_gpu_memory_gb:
                    continue

            # Filter by CPU and memory
            if inst.vcpus < req.min_vcpus:
                continue
            if inst.memory_gb < req.min_memory_gb:
                continue

            matches.append(inst)

        # Sort by price (cheapest first)
        matches.sort(key=lambda x: x.price_per_hour)

        return matches


class EC2Spawner:
    """Spawn EC2 instances with Docker image"""

    def __init__(self, region: str = 'us-east-1'):
        self.ec2 = boto3.client('ec2', region_name=region)
        self.region = region

    def spawn_instance(self,
                      instance_type: str,
                      image_name: str,
                      key_name: Optional[str] = None,
                      security_group: Optional[str] = None,
                      dry_run: bool = False) -> Optional[str]:
        """Spawn EC2 instance with Docker image"""

        # Get latest Amazon Linux 2 AMI with Docker
        ami_response = self.ec2.describe_images(
            Owners=['amazon'],
            Filters=[
                {'Name': 'name', 'Values': ['amzn2-ami-ecs-hvm-*-x86_64-ebs']},
                {'Name': 'state', 'Values': ['available']}
            ]
        )

        if not ami_response['Images']:
            print("Error: No suitable AMI found")
            return None

        # Get most recent AMI
        ami_id = sorted(ami_response['Images'],
                       key=lambda x: x['CreationDate'],
                       reverse=True)[0]['ImageId']

        print(f"Using AMI: {ami_id}")

        # User data script to pull and run Docker image
        user_data = f"""#!/bin/bash
yum update -y
yum install -y docker
service docker start
usermod -a -G docker ec2-user

# Pull Docker image
docker pull {image_name}

# Log completion
echo "Instance ready with image: {image_name}" > /var/log/instance-ready.log
"""

        # Instance configuration
        config = {
            'ImageId': ami_id,
            'InstanceType': instance_type,
            'MinCount': 1,
            'MaxCount': 1,
            'UserData': user_data,
            'TagSpecifications': [{
                'ResourceType': 'instance',
                'Tags': [
                    {'Key': 'Name', 'Value': f'docker-{image_name.split("/")[-1].split(":")[0]}'},
                    {'Key': 'ManagedBy', 'Value': 'ec2-optimizer'},
                    {'Key': 'DockerImage', 'Value': image_name}
                ]
            }]
        }

        if key_name:
            config['KeyName'] = key_name

        if security_group:
            config['SecurityGroupIds'] = [security_group]

        if dry_run:
            print("\n=== DRY RUN ===")
            print(f"Would launch: {instance_type}")
            print(f"AMI: {ami_id}")
            print(f"Docker image: {image_name}")
            print(f"User data:\n{user_data}")
            return None

        # Launch instance
        try:
            response = self.ec2.run_instances(**config, DryRun=False)
            instance_id = response['Instances'][0]['InstanceId']

            print(f"\n✓ Instance launched: {instance_id}")
            print(f"  Type: {instance_type}")
            print(f"  Region: {self.region}")
            print(f"  Docker image: {image_name}")
            print(f"\nWaiting for instance to start...")

            # Wait for instance to be running
            waiter = self.ec2.get_waiter('instance_running')
            waiter.wait(InstanceIds=[instance_id])

            # Get public IP
            instance_info = self.ec2.describe_instances(InstanceIds=[instance_id])
            public_ip = instance_info['Reservations'][0]['Instances'][0].get('PublicIpAddress', 'N/A')

            print(f"\n✓ Instance running!")
            print(f"  Instance ID: {instance_id}")
            print(f"  Public IP: {public_ip}")
            print(f"\nTo connect:")
            print(f"  aws ec2 describe-instances --instance-ids {instance_id}")
            if key_name:
                print(f"  ssh -i ~/.ssh/{key_name}.pem ec2-user@{public_ip}")

            return instance_id

        except Exception as e:
            print(f"Error launching instance: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description='EC2 Optimizer: Find and spawn optimal EC2 instances for Docker images')
    parser.add_argument('--image', required=True, help='Docker image name (e.g., pytorch/pytorch:latest)')
    parser.add_argument('--spawn', action='store_true', help='Actually spawn the instance (default: just recommend)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be launched without launching')
    parser.add_argument('--memory', type=int, help='Override minimum memory (GB)')
    parser.add_argument('--vcpus', type=int, help='Override minimum vCPUs')
    parser.add_argument('--gpu', choices=['required', 'optional', 'none'], help='GPU requirement override')
    parser.add_argument('--region', default='us-east-1', help='AWS region (default: us-east-1)')
    parser.add_argument('--key-name', help='SSH key pair name')
    parser.add_argument('--security-group', help='Security group ID')
    parser.add_argument('--top-n', type=int, default=5, help='Show top N recommendations (default: 5)')

    args = parser.parse_args()

    print("="*80)
    print("EC2 OPTIMIZER")
    print("="*80)
    print(f"\nAnalyzing Docker image: {args.image}\n")

    # Analyze Docker image
    analyzer = DockerImageAnalyzer()
    req = analyzer.analyze_image(args.image)

    # Apply overrides
    if args.memory:
        req.min_memory_gb = args.memory
    if args.vcpus:
        req.min_vcpus = args.vcpus
    if args.gpu == 'required':
        req.gpu_required = True
    elif args.gpu == 'none':
        req.gpu_required = False

    print(f"\n{req}\n")

    # Find matching instances
    selector = EC2InstanceSelector()
    matches = selector.select_instance(req)

    if not matches:
        print("❌ No matching instances found!")
        print("Try relaxing requirements or use --memory/--vcpus overrides")
        sys.exit(1)

    # Show recommendations
    print("="*80)
    print(f"TOP {min(args.top_n, len(matches))} RECOMMENDATIONS (Sorted by price)")
    print("="*80)

    for i, inst in enumerate(matches[:args.top_n], 1):
        print(f"\n{i}. {inst}")
        print(f"   Monthly cost (730 hrs): ${inst.monthly_cost():.2f}")
        print(f"   Daily cost (24 hrs): ${inst.price_per_hour * 24:.2f}")

    # Spawn instance if requested
    if args.spawn or args.dry_run:
        best = matches[0]
        print("\n" + "="*80)
        print(f"{'DRY RUN: Would launch' if args.dry_run else 'Launching'}: {best.instance_type}")
        print("="*80)

        spawner = EC2Spawner(region=args.region)
        instance_id = spawner.spawn_instance(
            instance_type=best.instance_type,
            image_name=args.image,
            key_name=args.key_name,
            security_group=args.security_group,
            dry_run=args.dry_run
        )

        if instance_id:
            print(f"\n✓ Success! Instance {instance_id} is running.")
    else:
        print("\n" + "="*80)
        print("To launch the best option:")
        print("="*80)
        print(f"python {sys.argv[0]} --image {args.image} --spawn")
        if args.key_name:
            print(f"  --key-name {args.key_name}")
        if args.security_group:
            print(f"  --security-group {args.security_group}")
        print()


if __name__ == '__main__':
    main()
