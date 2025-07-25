# AWS EC2 Deployment Guide for CV-SAR SR Pipeline

## 🚀 Overview
Complete guide for deploying the CV-SAR SR patch extraction pipeline on AWS EC2 with GPU support, auto-scaling, and S3 integration.

## 📋 Prerequisites
- AWS Account with EC2 and S3 access
- Docker and Docker Compose installed locally
- AWS CLI configured
- ECR (Elastic Container Registry) access

## 🏗️ Architecture
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   S3 Input  │────▶│  EC2 GPU    │────▶│  S3 Output  │
│   Buckets   │     │  Instances  │     │   Buckets   │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                    ┌──────▼──────┐
                    │ Auto Scaling │
                    │    Group     │
                    └──────────────┘
```

## 1️⃣ EC2 Instance Setup

### Recommended Instance Types
```bash
# For GPU acceleration (recommended)
- p3.2xlarge  # 1x V100 GPU, 8 vCPU, 61 GB RAM
- g4dn.xlarge # 1x T4 GPU, 4 vCPU, 16 GB RAM (cost-effective)
- g5.2xlarge  # 1x A10G GPU, 8 vCPU, 32 GB RAM

# For CPU-only (budget option)
- c5.4xlarge  # 16 vCPU, 32 GB RAM
- m5.8xlarge  # 32 vCPU, 128 GB RAM
```

### Launch EC2 Instance
```bash
# Using AWS CLI
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \  # Deep Learning AMI
  --instance-type g4dn.xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxx \
  --subnet-id subnet-xxxxxxxx \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":100,\"VolumeType\":\"gp3\"}}]" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=cv-sar-sr-gpu}]'
```

## 2️⃣ Docker Image Build & Push

### Build Docker Image
```bash
# Clone repository
git clone <your-repo>
cd Sentinel-1

# Build image
docker build -f workflows/Dockerfile -t cv-sar-sr:latest .

# Test locally
docker run --rm --gpus all cv-sar-sr:latest test
```

### Push to ECR
```bash
# Create ECR repository
aws ecr create-repository --repository-name cv-sar-sr

# Get login token
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  123456789012.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag cv-sar-sr:latest \
  123456789012.dkr.ecr.us-east-1.amazonaws.com/cv-sar-sr:latest

docker push \
  123456789012.dkr.ecr.us-east-1.amazonaws.com/cv-sar-sr:latest
```

## 3️⃣ S3 Bucket Setup

### Create S3 Buckets
```bash
# Input bucket
aws s3 mb s3://cv-sar-sr-input-bucket
aws s3api put-bucket-versioning \
  --bucket cv-sar-sr-input-bucket \
  --versioning-configuration Status=Enabled

# Output bucket
aws s3 mb s3://cv-sar-sr-output-bucket

# Upload input data
aws s3 sync ./data/processed_1/ \
  s3://cv-sar-sr-input-bucket/processed_1/ \
  --exclude "*" --include "*.dim" --include "*.data/*"
```

### IAM Role for EC2
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::cv-sar-sr-input-bucket/*",
        "arn:aws:s3:::cv-sar-sr-input-bucket"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::cv-sar-sr-output-bucket/*",
        "arn:aws:s3:::cv-sar-sr-output-bucket"
      ]
    }
  ]
}
```

## 4️⃣ Deploy on EC2

### SSH to EC2 Instance
```bash
ssh -i your-key.pem ec2-user@<instance-public-ip>
```

### Install Docker (if needed)
```bash
# Update system
sudo yum update -y

# Install Docker
sudo amazon-linux-extras install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
  -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install NVIDIA Container Toolkit (for GPU)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Run Pipeline
```bash
# Create .env file
cat > .env << EOF
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
S3_BUCKET=cv-sar-sr-input-bucket
S3_INPUT_PREFIX=processed_1
S3_OUTPUT_PREFIX=output/$(date +%Y%m%d_%H%M%S)
CLEANUP_AFTER_UPLOAD=true
EOF

# Pull image from ECR
docker pull 123456789012.dkr.ecr.us-east-1.amazonaws.com/cv-sar-sr:latest

# Run with docker-compose
docker-compose -f workflows/docker-compose.yml up cv-sar-sr-production
```

## 5️⃣ Auto Scaling Setup

### Create Launch Template
```bash
aws ec2 create-launch-template \
  --launch-template-name cv-sar-sr-template \
  --launch-template-data '{
    "ImageId": "ami-0c02fb55956c7d316",
    "InstanceType": "g4dn.xlarge",
    "IamInstanceProfile": {
      "Name": "cv-sar-sr-role"
    },
    "UserData": "'$(base64 -w 0 user-data.sh)'"
  }'
```

### User Data Script (user-data.sh)
```bash
#!/bin/bash
# Install dependencies
yum update -y
amazon-linux-extras install docker -y
service docker start

# Pull and run container
$(aws ecr get-login --no-include-email --region us-east-1)
docker pull 123456789012.dkr.ecr.us-east-1.amazonaws.com/cv-sar-sr:latest

# Run with environment from instance metadata
INSTANCE_ID=$(ec2-metadata --instance-id | cut -d " " -f 2)
docker run --rm --gpus all \
  -e AWS_DEFAULT_REGION=us-east-1 \
  -e S3_BUCKET=cv-sar-sr-input-bucket \
  -e S3_INPUT_PREFIX=processed_1 \
  -e S3_OUTPUT_PREFIX=output/${INSTANCE_ID}/$(date +%Y%m%d_%H%M%S) \
  -e CLEANUP_AFTER_UPLOAD=true \
  123456789012.dkr.ecr.us-east-1.amazonaws.com/cv-sar-sr:latest

# Shutdown after completion
shutdown -h now
```

### Create Auto Scaling Group
```bash
aws autoscaling create-auto-scaling-group \
  --auto-scaling-group-name cv-sar-sr-asg \
  --launch-template LaunchTemplateName=cv-sar-sr-template \
  --min-size 0 \
  --max-size 10 \
  --desired-capacity 2 \
  --vpc-zone-identifier subnet-xxxxxxxx
```

## 6️⃣ Monitoring & Logging

### CloudWatch Metrics
```bash
# Create CloudWatch dashboard
aws cloudwatch put-dashboard \
  --dashboard-name CV-SAR-SR-Monitor \
  --dashboard-body file://dashboard.json
```

### View Logs
```bash
# From EC2 instance
docker logs cv-sar-sr-prod

# From CloudWatch (if configured)
aws logs tail /aws/ec2/cv-sar-sr --follow
```

## 7️⃣ Cost Optimization

### Spot Instances
```bash
# Request spot instances for cost savings
aws ec2 request-spot-instances \
  --spot-price "0.30" \
  --instance-count 2 \
  --type "one-time" \
  --launch-specification file://spot-spec.json
```

### Lambda Trigger for Batch Processing
```python
# lambda_function.py
import boto3

def lambda_handler(event, context):
    # Triggered by S3 upload
    s3_event = event['Records'][0]['s3']
    bucket = s3_event['bucket']['name']
    key = s3_event['object']['key']
    
    # Start EC2 instance or ECS task
    ec2 = boto3.client('ec2')
    response = ec2.start_instances(
        InstanceIds=['i-1234567890abcdef0']
    )
    
    return {
        'statusCode': 200,
        'body': f'Started processing {key}'
    }
```

## 8️⃣ Production Checklist

- [ ] ECR repository created and image pushed
- [ ] S3 buckets created with proper permissions
- [ ] IAM roles configured
- [ ] EC2 instance launched with GPU
- [ ] Docker and NVIDIA toolkit installed
- [ ] Environment variables configured
- [ ] Auto-scaling configured (optional)
- [ ] CloudWatch monitoring enabled
- [ ] Cost alerts set up

## 🎯 Quick Start Commands

```bash
# 1. Build and push image
docker build -f workflows/Dockerfile -t cv-sar-sr:latest .
aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_URI
docker tag cv-sar-sr:latest $ECR_URI/cv-sar-sr:latest
docker push $ECR_URI/cv-sar-sr:latest

# 2. Launch EC2 instance
aws ec2 run-instances --image-id ami-xxx --instance-type g4dn.xlarge

# 3. SSH and run
ssh -i key.pem ec2-user@instance-ip
docker run --rm --gpus all -e S3_BUCKET=xxx $ECR_URI/cv-sar-sr:latest

# 4. Check results
aws s3 ls s3://cv-sar-sr-output-bucket/output/
```

## 🆘 Troubleshooting

### GPU not detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Reinstall NVIDIA Container Toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### S3 permission denied
```bash
# Check IAM role
aws sts get-caller-identity

# Test S3 access
aws s3 ls s3://cv-sar-sr-input-bucket/
```

### Out of memory
```bash
# Reduce batch size
docker run -e MAX_WORKERS=2 ...

# Use larger instance
aws ec2 modify-instance-attribute --instance-id i-xxx --instance-type m5.8xlarge
```

## 📊 Performance Metrics

| Instance Type | GPU | vCPU | RAM | Patches/Hour | Cost/Hour |
|--------------|-----|------|-----|--------------|-----------|
| g4dn.xlarge  | T4  | 4    | 16  | ~2,000       | $0.526    |
| p3.2xlarge   | V100| 8    | 61  | ~5,000       | $3.06     |
| c5.4xlarge   | -   | 16   | 32  | ~800         | $0.68     |

## 🎉 Success!
Your CV-SAR SR pipeline is now running on AWS EC2 with GPU acceleration! 