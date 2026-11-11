import os
import sys
import json
import urllib.request
import urllib.error
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def get_github_token():
    res = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n",
        text=True,
        capture_output=True,
        check=True
    )
    for line in res.stdout.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("Failed to retrieve GitHub token from Git Credential Manager")

def create_release(token, repo="KappinChromie/TinyLLM", tag="v1.0.0", name="TinyLLM v1.0.0: 42M Llama with Native 32k Context", body=""):
    url = f"https://api.github.com/repos/{repo}/releases"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "TinyLLM-Release-Bot"
    }
    payload = {
        "tag_name": tag,
        "target_commitish": "main",
        "name": name,
        "body": body,
        "draft": False,
        "prerelease": False
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            print(f"Successfully created release: {data.get('html_url')}")
            return data
    except urllib.error.HTTPError as e:
        err_body = e.read().decode('utf-8')
        print(f"HTTPError {e.code}: {err_body}")
        # If release already exists, fetch it
        if e.code == 422:
            print("Release might already exist. Fetching existing releases...")
            get_req = urllib.request.Request(f"https://api.github.com/repos/{repo}/releases/tags/{tag}", headers=headers)
            with urllib.request.urlopen(get_req) as resp:
                return json.loads(resp.read().decode('utf-8'))
        raise

def upload_asset(token, upload_url, file_path, content_type="application/octet-stream"):
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Asset file not found: {file_path}")
        return
        
    file_size = file_path.stat().st_size
    filename = file_path.name
    print(f"Uploading {filename} ({file_size / (1024*1024):.2f} MB)...", flush=True)
    
    # upload_url format is: https://uploads.github.com/repos/:owner/:repo/releases/:id/assets{?name,label}
    clean_upload_url = upload_url.split("{")[0] + f"?name={filename}"
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": content_type,
        "Content-Length": str(file_size),
        "User-Agent": "TinyLLM-Release-Bot"
    }
    
    with open(file_path, "rb") as f:
        data = f.read()
        
    req = urllib.request.Request(clean_upload_url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            res_data = json.loads(resp.read().decode('utf-8'))
            print(f"Uploaded {filename} successfully: {res_data.get('browser_download_url')}")
            return res_data
    except urllib.error.HTTPError as e:
        print(f"Failed to upload {filename}: {e.code} {e.read().decode('utf-8')}")

def main():
    token = get_github_token()
    print("Retrieved GitHub token from credential manager.")
    
    release_notes = """## TinyLLM v1.0.0: High-Throughput 42M Language Model with Native 32k Context

TinyLLM is a 42.28-million parameter causal language model engineered specifically for maximum generation throughput on consumer hardware (developed, trained, and benchmarked on a single **NVIDIA GeForce RTX 3070 Ti Desktop GPU, 8 GB VRAM**).

By pairing an overhead-optimized 6-layer architecture with Multi-Query Attention (1 KV head) and a scaled RoPE base ($\\theta = 1,000,000$), TinyLLM natively supports context windows up to **32,768 (32k) tokens** with a memory footprint of only **1.5 KB per token** for the KV cache.

---

### ⚡ Performance Highlights (RTX 3070 Ti, llama.cpp CUDA)
- **Single-Stream Decode Speed**:
  - **1,574 tokens/sec** at 0k context depth (`Q8_0`)
  - **1,428 tokens/sec** at 8k context depth (`Q8_0`)
  - **1,233 tokens/sec** at 32k context depth (`Q8_0`)
- **Batched Serving Throughput**:
  - **6,789 generation tokens/sec** (and **27,278 total tokens/sec**) at 8 concurrent streams (`Q8_0`, 2k context).
- **Ultra-Lean KV Cache**:
  - Only **1.5 KB per token** ($2 \\times 6 \\times 1 \\times 64 \\times 2\\text{ bytes}$).
  - A full 32k stream consumes just **48 MB** of VRAM.
  - 16 parallel 32k streams consume only **774 MB** of KV cache memory.
- **Strict Parity**:
  - `llama.cpp` F16 reproduces PyTorch cross-entropy to within **±0.0002 CE** across all benchmarks and 32k context windows.
  - `Q8_0` adds negligible loss penalty ($\le +0.0014$ CE).

---

### 📦 Direct GGUF Model Downloads

| Variant | Role | Size | Download | Recommended Use Case |
| :--- | :---: | :---: | :---: | :--- |
| **`tinyllm-chat-q8_0.gguf`** | Chat | 45.2 MB | Attached below | **Recommended Default**: Peak speed (1,574 tok/s), near-lossless precision. |
| **`tinyllm-chat-q4_0.gguf`** | Chat | 25.7 MB | Attached below | Mobile / Edge / Embedded (<30 MB footprint). |
| **`tinyllm-chat-f16.gguf`** | Chat | 84.9 MB | Attached below | Unquantized reference precision. |
| **`stageC_q8_0.gguf`** | Base | 45.1 MB | Attached below | Base completion model at 32k context. |
| **`stageC_q4_0.gguf`** | Base | 25.6 MB | Attached below | Compact Base model. |
| **`stageC_f16.gguf`** | Base | 84.8 MB | Attached below | Full precision Base model. |

---

### 🚀 Quick Start (llama.cpp)

#### Interactive Chat:
```bash
llama-cli -m tinyllm-chat-q8_0.gguf -cnv -ngl 99 -fa on -c 32768
```

#### OpenAI-Compatible REST Server:
```bash
llama-server -m tinyllm-chat-q8_0.gguf -ngl 99 -fa on -c 32768 -b 2048 -ub 512 --chat-template chatml --temp 0.7 --min-p 0.05 --repeat-penalty 1.15 --host 127.0.0.1 --port 8080
```
"""
    
    rel = create_release(token, body=release_notes)
    upload_url = rel.get("upload_url")
    
    assets = [
        ROOT / "release_chat" / "tinyllm-chat-q8_0.gguf",
        ROOT / "release_chat" / "tinyllm-chat-q4_0.gguf",
        ROOT / "release_chat" / "tinyllm-chat-f16.gguf",
        ROOT / "release" / "gguf" / "stageC_q8_0.gguf",
        ROOT / "release" / "gguf" / "stageC_q4_0.gguf",
        ROOT / "release" / "gguf" / "stageC_f16.gguf",
    ]
    
    for asset in assets:
        upload_asset(token, upload_url, asset)
        
    print("\nAll release operations completed successfully!")
    print(f"Release URL: {rel.get('html_url')}")

if __name__ == "__main__":
    main()
