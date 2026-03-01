# Connect Azure VM to Laptop GLiNER Server

## Status
- [x] Laptop: Tailscale installed and connected (IP: `100.67.243.94`)
- [x] Laptop: `gliner-inference-server` running on GPU (port 8081)
- [ ] Azure VM: Install Tailscale
- [ ] Azure VM: Update `.env` with laptop IP
- [ ] End-to-end test

## Step 1: Install Tailscale on Azure VM

SSH into the Azure VM and run:

```bash
curl -fsSL https://tailscale.com/install.sh | sudo sh
sudo tailscale up
```

Authenticate with the **same Tailscale account** used on the laptop. After login:

```bash
# Verify connectivity from Azure VM to laptop
curl http://100.67.243.94:8081/health
# Expected: {"status":"ok","model":"nvidia/gliner-PII","device":"cuda"}
```

## Step 2: Verify `.env`

The `.env` in the Mistral repo already contains:

```
GLINER_SERVER_URL=http://100.67.243.94:8081
```

No changes needed — just make sure it's deployed to Azure.

## Step 3: Test the pipeline

1. Start the inference server on laptop (if not already running):
   ```bash
   cd ~/coding/gliner-inference-server
   python server.py
   ```

2. Run a pipeline from the frontend — SSE should show the scanning phase with detected PII categories.

3. Stop the inference server, run the pipeline again — scanning should be skipped gracefully with "inference server unreachable".

## Notes
- Laptop Tailscale IP: `100.67.243.94` (stable, won't change)
- If the laptop is off or server not running, the backend skips PII scanning automatically
- The inference server runs on `0.0.0.0:8081` so it accepts connections from any interface including Tailscale
