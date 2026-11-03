import os
import subprocess
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
TOOLS_DIR = ROOT / "tools"
LLAMA_BIN = TOOLS_DIR / "llama-bin"
QUANTIZE_BIN = LLAMA_BIN / "llama-quantize.exe"
CONVERT_SCRIPT = TOOLS_DIR / "llama.cpp-src" / "convert_hf_to_gguf.py"

def download_and_convert(repo_id: str, short_name: str):
    print(f"\n=== Processing {repo_id} ({short_name}) ===")
    hf_dir = MODELS_DIR / f"hf_{short_name}"
    print(f"Downloading from Hugging Face: {repo_id} -> {hf_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(hf_dir),
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
    )
    
    f16_gguf = MODELS_DIR / f"{short_name}-f16.gguf"
    print(f"Converting to GGUF F16: {f16_gguf}")
    cmd_convert = [
        sys.executable,
        str(CONVERT_SCRIPT),
        str(hf_dir),
        "--outfile", str(f16_gguf),
        "--outtype", "f16"
    ]
    subprocess.run(cmd_convert, check=True)
    
    for quant in ["Q8_0", "Q4_0"]:
        quant_gguf = MODELS_DIR / f"{short_name}-{quant}.gguf"
        print(f"Quantizing to {quant}: {quant_gguf}")
        cmd_quant = [
            str(QUANTIZE_BIN),
            str(f16_gguf),
            str(quant_gguf),
            quant
        ]
        subprocess.run(cmd_quant, check=True)

if __name__ == "__main__":
    download_and_convert("HuggingFaceTB/SmolLM2-135M", "smollm2-135m")
    download_and_convert("Qwen/Qwen3-0.6B", "qwen3-0.6b")
    print("\nAll baseline models prepared successfully!")
