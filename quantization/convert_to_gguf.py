import os
import subprocess
import argparse
import sys

def clone_llama_cpp():
    if not os.path.exists("llama.cpp"):
        print("Cloning llama.cpp repository...")
        subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp"], check=True)
        print("Building llama.cpp...")
        subprocess.run(["make", "-C", "llama.cpp"], check=True)
        # Install requirements
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "llama.cpp/requirements.txt"], check=True)
    else:
        print("llama.cpp already exists. Skipping clone.")

def convert_to_gguf(model_path, output_path, out_type="f16"):
    print(f"Converting {model_path} to GGUF ({out_type})...")
    
    # Run convert-hf-to-gguf.py
    convert_script = os.path.join("llama.cpp", "convert_hf_to_gguf.py")
    if not os.path.exists(convert_script):
        # newer llama.cpp uses convert-hf-to-gguf.py (underscores) or convert.py
        # Check for convert.py as fallback
        convert_script = os.path.join("llama.cpp", "convert.py")
    
    if not os.path.exists(convert_script):
        print("Error: Could not find conversion script in llama.cpp")
        return

    cmd = [
        sys.executable,
        convert_script,
        model_path,
        "--outfile", output_path,
        "--outtype", out_type
    ]
    
    subprocess.run(cmd, check=True)
    print(f"Conversion complete: {output_path}")

def quantize_gguf(input_gguf, output_gguf, method="Q4_K_M"):
    print(f"Quantizing {input_gguf} to {method}...")
    
    quantize_bin = os.path.join("llama.cpp", "llama-quantize")
    if not os.path.exists(quantize_bin):
        print("Error: llama-quantize binary not found. Did build fail?")
        return

    cmd = [
        quantize_bin,
        input_gguf,
        output_gguf,
        method
    ]
    
    subprocess.run(cmd, check=True)
    print(f"Quantization complete: {output_gguf}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert and Quantize model to GGUF")
    parser.add_argument("--model_path", type=str, required=True, help="Path to merged HF model")
    parser.add_argument("--output_dir", type=str, default="./models", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Setup llama.cpp
    clone_llama_cpp()
    
    # 2. Convert to FP16 GGUF first
    fp16_output = os.path.join(args.output_dir, "sbicard-slm-f16.gguf")
    convert_to_gguf(args.model_path, fp16_output, "f16")
    
    # 3. Quantize to Q4_K_M (Recommended for 4-bit)
    final_output = os.path.join(args.output_dir, "sbicard-slm.gguf")
    quantize_gguf(fp16_output, final_output, "Q4_K_M")
    
    # Cleanup big fp16 file if needed
    # os.remove(fp16_output)
    print(f"Final model ready at {final_output}")
