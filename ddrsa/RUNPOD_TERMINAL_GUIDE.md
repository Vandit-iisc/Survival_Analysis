# RunPod Terminal Guide - Complete Walkthrough

## Step-by-Step: From GitHub to Running on GPU

### Part 1: Create a Pod (Not Endpoint)

**Important**: For training, you need a **Pod**, not an **Endpoint**!
- **Pods**: For development, training, interactive work ✅
- **Endpoints**: For deployment, API serving (not what you need)

#### 1.1 Navigate to Pods

1. Go to https://runpod.io
2. Click **"Pods"** in the left sidebar (not Serverless/Endpoints)

#### 1.2 Deploy a New Pod

1. Click **"+ Deploy"** or **"Deploy Pod"**
2. Select GPU:
   - **Recommended**: RTX 3080 (10GB)
   - Filter by: "RTX 3080"
   - Choose "On-Demand" or "Spot" (Spot is cheaper but can be interrupted)

#### 1.3 Configure Pod

1. **Template**: Select **"RunPod Pytorch"** or **"RunPod Tensorflow"**
   - Or select: **"Ubuntu 22.04 + CUDA 11.8"**

2. **Storage**:
   - Container Disk: 20 GB (minimum)
   - Volume Disk: 30 GB (optional, for persistent storage)

3. **Expose Ports** (click "Edit"):
   - Add port: `6006` (for TensorBoard)
   - Add port: `8888` (for Jupyter, optional)

4. Click **"Deploy On-Demand"** or **"Deploy Spot"**

### Part 2: Access Terminal

#### Option A: Web Terminal (Easiest)

1. Once pod is deployed, you'll see it in "My Pods"
2. Click **"Connect"** button
3. Select **"Start Web Terminal"** or **"Connect to Jupyter Lab"**
4. A terminal window opens in your browser ✅

#### Option B: SSH (More Powerful)

1. Click **"Connect"** → **"TCP Port Mappings"**
2. Copy the SSH command:
   ```bash
   ssh root@<pod-id>.runpod.io -p <port>
   ```
3. Or use the direct SSH command shown
4. Paste in your local terminal

#### Option C: Jupyter Lab Terminal

1. Click **"Connect"** → **"Connect to Jupyter Lab"**
2. In Jupyter: **File → New → Terminal**

### Part 3: Get Your Code on RunPod

You mentioned you linked GitHub - great! Here are the methods:

#### Method 1: Clone from GitHub (Recommended)

```bash
# In RunPod terminal:
cd /workspace

# Clone your repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# If private repo, you may need to authenticate
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# For private repos, use SSH key or personal access token
```

#### Method 2: Upload Files via Jupyter

1. Connect to Jupyter Lab
2. Click **Upload** button (top left)
3. Select your files/folders
4. Upload

#### Method 3: Upload via RunPod Interface

1. In Pod view, click **"Connect"**
2. Use **file browser** if available
3. Drag and drop files

#### Method 4: Use Git (If code is already there)

If you used RunPod's GitHub integration:
```bash
cd /workspace
ls  # Your code might already be there
```

### Part 4: Setup Environment

```bash
# Navigate to your code
cd /workspace/ddrsa  # or wherever your code is

# Check Python version
python --version  # Should be 3.8+

# Install dependencies
pip install torch numpy pandas scikit-learn tqdm tensorboard matplotlib

# OR use requirements.txt
pip install -r requirements.txt
```

### Part 5: Upload Dataset

#### Option 1: Direct Upload

```bash
# Create data directory
mkdir -p /workspace/Challenge_Data

# Then upload files via Jupyter interface to this folder
```

#### Option 2: Download from Cloud

If your data is on Google Drive, Dropbox, etc.:

```bash
# Example with wget
wget "https://your-data-url.com/data.zip" -O data.zip
unzip data.zip -d Challenge_Data/

# Example with Google Drive (requires gdown)
pip install gdown
gdown "https://drive.google.com/uc?id=YOUR_FILE_ID"
```

#### Option 3: Copy from Local via SCP

From your local machine:
```bash
# Get SSH command from RunPod
scp -P <port> -r Challenge_Data/ root@<pod-id>.runpod.io:/workspace/
```

### Part 6: Verify GPU

```bash
# Check GPU is available
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.x.x    Driver Version: 525.x.x    CUDA Version: 11.8       |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# |   0  NVIDIA GeForce RTX 3080                           |

# Check PyTorch can see GPU
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0)}')"

# Expected output:
# GPU Available: True
# GPU Name: NVIDIA GeForce RTX 3080
```

### Part 7: Test Your Code

```bash
# Run test
python test_installation.py

# Expected output:
# ✓ GPU detected: NVIDIA GeForce RTX 3080
# ✓ All tests passed!
```

### Part 8: Run Training

#### Quick Test (5 epochs, 2-3 minutes)

```bash
python main.py \
    --model-type rnn \
    --num-epochs 5 \
    --batch-size 64 \
    --exp-name gpu_quick_test
```

#### Monitor GPU Usage

Open a second terminal (or use `screen`):
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or
nvtop  # if available (more visual)
```

#### Full Training (100 epochs, 20-25 minutes)

```bash
python main.py \
    --model-type rnn \
    --num-epochs 100 \
    --batch-size 128 \
    --exp-name ddrsa_rnn \
    --data-path /workspace/Challenge_Data
```

### Part 9: Run in Background (Important!)

If training takes long, run in background so you can close browser:

#### Option 1: Using `nohup`

```bash
nohup python main.py \
    --model-type rnn \
    --num-epochs 100 \
    --exp-name ddrsa_rnn \
    > training.log 2>&1 &

# Check progress
tail -f training.log

# Check if still running
ps aux | grep python
```

#### Option 2: Using `screen` (Better)

```bash
# Install screen
apt-get update && apt-get install -y screen

# Start a screen session
screen -S training

# Run your training
python main.py --num-epochs 100 --exp-name ddrsa_rnn

# Detach from screen: Press Ctrl+A, then D
# Your training continues in background!

# Reattach later
screen -r training

# List all sessions
screen -ls

# Kill a session
screen -X -S training quit
```

#### Option 3: Using `tmux` (Alternative to screen)

```bash
# Install tmux
apt-get update && apt-get install -y tmux

# Start tmux
tmux new -s training

# Run training
python main.py --num-epochs 100

# Detach: Press Ctrl+B, then D

# Reattach
tmux attach -t training
```

### Part 10: Monitor Training

#### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir logs/ --host 0.0.0.0 --port 6006

# Access in browser:
# 1. Go to RunPod pod page
# 2. Click "Connect" → "HTTP Service"
# 3. Port 6006 should be listed
# 4. Click the link
```

#### Check Progress

```bash
# View logs
tail -f training.log  # if using nohup

# Check current epoch
ls -lt logs/ddrsa_rnn/checkpoints/

# View metrics
cat logs/ddrsa_rnn/test_metrics.json
```

### Part 11: Download Results

#### Option 1: Via Jupyter Interface

1. Go to Jupyter Lab
2. Navigate to `logs/` or `figures/`
3. Right-click → **Download**

#### Option 2: Create Archive and Download

```bash
# Create archive
tar -czf results.tar.gz logs/ figures/

# Download via Jupyter or use SCP from local:
scp -P <port> root@<pod-id>.runpod.io:/workspace/results.tar.gz ./
```

#### Option 3: Push to GitHub

```bash
# Commit results
git add logs/ figures/
git commit -m "Training results"
git push origin main

# Then pull on your local machine
```

#### Option 4: Upload to Cloud

```bash
# Google Drive (using rclone)
apt-get install -y rclone
rclone config  # Configure Google Drive
rclone copy logs/ gdrive:ddrsa_results/

# Or use AWS S3, etc.
```

### Part 12: Stop Pod (Important!)

**Don't forget to stop your pod when done!**

1. Go to RunPod dashboard
2. Find your pod
3. Click **"Stop"** or **"Terminate"**
4. Confirm

**Cost savings**: Stopped pods don't incur charges!

### Part 13: Resume Later (If using Volume Storage)

If you set up volume storage:

1. Deploy new pod
2. Attach same volume
3. Your code and data are still there!
4. Continue training:
   ```bash
   cd /workspace/ddrsa
   python main.py --exp-name resume_training
   ```

## Complete Example Workflow

Here's a complete session from start to finish:

```bash
# ============================================
# 1. SETUP (First time only)
# ============================================

# Navigate to workspace
cd /workspace

# Clone your code
git clone https://github.com/YOUR_USERNAME/ddrsa.git
cd ddrsa

# Install dependencies
pip install -r requirements.txt

# Upload data (via Jupyter or download)
mkdir -p /workspace/Challenge_Data
# ... upload train.txt, test.txt, etc.

# ============================================
# 2. VERIFY SETUP
# ============================================

# Check GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Test code
python test_installation.py

# ============================================
# 3. QUICK TEST
# ============================================

python main.py \
    --model-type rnn \
    --num-epochs 5 \
    --batch-size 64 \
    --exp-name quick_test \
    --data-path /workspace/Challenge_Data

# ============================================
# 4. FULL TRAINING (in background)
# ============================================

# Start screen session
screen -S training

# Run full training
python main.py \
    --model-type rnn \
    --num-epochs 100 \
    --batch-size 128 \
    --exp-name ddrsa_rnn \
    --data-path /workspace/Challenge_Data

# Detach: Ctrl+A, then D

# ============================================
# 5. MONITOR (in another terminal)
# ============================================

# Watch GPU
watch -n 1 nvidia-smi

# Start TensorBoard
tensorboard --logdir logs/ --host 0.0.0.0 --port 6006

# ============================================
# 6. AFTER TRAINING
# ============================================

# Reattach to screen
screen -r training

# Create visualizations
python create_figures.py --exp-name ddrsa_rnn

# Package results
tar -czf results.tar.gz logs/ figures/

# Download via Jupyter or SCP

# ============================================
# 7. CLEANUP
# ============================================

# Stop pod via RunPod dashboard!
```

## Troubleshooting

### "Command not found"

```bash
# Update package list
apt-get update

# Install missing packages
apt-get install -y screen tmux wget unzip git
```

### "Permission denied"

You're root by default on RunPod, so this shouldn't happen. But if it does:
```bash
chmod +x your_script.sh
```

### "No space left on device"

```bash
# Check disk usage
df -h

# Clean up
rm -rf ~/.cache/pip
apt-get clean

# Or increase container disk size when deploying pod
```

### "CUDA out of memory"

```bash
# Reduce batch size
python main.py --batch-size 32  # or 16
```

### Can't access TensorBoard

1. Make sure port 6006 is exposed in pod settings
2. Use `--host 0.0.0.0` when starting TensorBoard
3. Check RunPod "Connect" → "HTTP Service" for the URL

## Summary

1. **Create Pod** (not Endpoint) - Select RTX 3080
2. **Connect** → Web Terminal or SSH
3. **Clone code**: `git clone ...`
4. **Install deps**: `pip install -r requirements.txt`
5. **Upload data**: Via Jupyter or download
6. **Test**: `python test_installation.py`
7. **Train**: `python main.py --num-epochs 100`
8. **Monitor**: `nvidia-smi`, TensorBoard
9. **Download results**: Via Jupyter or SCP
10. **Stop pod**: Don't forget!

Your code will run on GPU automatically if `torch.cuda.is_available()` returns `True`!
