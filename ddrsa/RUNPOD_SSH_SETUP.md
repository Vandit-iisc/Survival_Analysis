# RunPod SSH Setup Guide

## Problem
Getting `Permission denied (publickey)` when trying to SSH to RunPod.

## Solution 1: Use Web Terminal (Recommended for Beginners)

**No SSH setup required!**

1. Go to https://runpod.io
2. Navigate to "My Pods"
3. Find your Pod
4. Click **"Connect"**
5. Select **"Start Web Terminal"**
6. Terminal opens in browser ✅

## Solution 2: Set Up SSH Keys (For Advanced Users)

### Step 1: Generate SSH Key on Your Local Machine

```bash
# Generate new SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# When prompted for file location, press Enter to accept default
# When prompted for passphrase, you can press Enter for no passphrase
```

This creates two files:
- `~/.ssh/id_ed25519` (private key - keep secret!)
- `~/.ssh/id_ed25519.pub` (public key - share with RunPod)

### Step 2: Copy Your Public Key

```bash
# Display your public key
cat ~/.ssh/id_ed25519.pub
```

Copy the entire output (starts with `ssh-ed25519 ...`)

### Step 3: Add SSH Key to RunPod

1. Go to https://runpod.io
2. Click your profile icon (top right)
3. Go to **"Settings"**
4. Navigate to **"SSH Keys"** section
5. Click **"Add SSH Key"**
6. Paste your public key
7. Give it a name (e.g., "My MacBook")
8. Click **"Save"**

### Step 4: Connect via SSH

Now you can connect:

```bash
# Use the SSH command from RunPod's "Connect" menu
ssh root@<pod-id>-<port>.ssh.runpod.io -p <port>

# Or use the full command shown in RunPod dashboard
```

### Step 5: Add to SSH Config (Optional - For Convenience)

Create/edit `~/.ssh/config`:

```bash
nano ~/.ssh/config
```

Add this:

```
Host runpod
    HostName <pod-id>-<port>.ssh.runpod.io
    User root
    Port <port>
    IdentityFile ~/.ssh/id_ed25519
```

Now you can connect with just:
```bash
ssh runpod
```

## Troubleshooting

### "Permission denied (publickey)"
- Make sure you added your **public key** (`id_ed25519.pub`) to RunPod, not the private key
- Wait a few minutes after adding the key to RunPod
- Restart your Pod if needed

### "Identity file not accessible"
- You need to generate the SSH key first (Step 1 above)
- Make sure the file exists: `ls -la ~/.ssh/`

### Still Can't Connect?
- **Use Web Terminal instead!** It's easier and works immediately
- Or use Jupyter Lab → File → New → Terminal

## Recommendation

For getting started with DDRSA training, I strongly recommend using the **Web Terminal**. You can always set up SSH later if you need it for file transfers or prefer working in your local terminal.

## Quick Start with Web Terminal

Once you're in the Web Terminal:

```bash
# Navigate to workspace
cd /workspace

# Clone your code
git clone https://github.com/YOUR_USERNAME/ddrsa.git
cd ddrsa

# Install dependencies
pip install -r requirements.txt

# Verify GPU
nvidia-smi
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Test installation
python test_installation.py

# Start training!
python main.py --num-epochs 100 --batch-size 128 --exp-name ddrsa_rnn
```

You're ready to go!
