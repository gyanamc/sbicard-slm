import os
import argparse
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

def quantize(model_path, output_path, quant_config):
    print(f"Loading model from {model_path}...")
    model = AutoAWQForCausalLM.from_pretrained(
        model_path, 
        **{"low_cpu_mem_usage": True, "use_cache": False}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print("Quantizing model...")
    # We need calibration data. For simplicity, we can use the model itself to generate some,
    # or pass a subset of the training data. AutoAWQ handles this.
    # Ideally, we pass 'calib_data' which is a list of text samples.
    # Here we assume no calibration data provided for simplicity, relying on default data or none,
    # but strictly AWQ usually needs calibration.
    # Let's add a placeholder for calibration data loading if needed, or stick to basic usage.
    # For now, following basic AutoAWQ usage without explicit calib data for brevity, 
    # but in production, we should load our dataset.
    
    model.quantize(tokenizer, quant_config=quant_config)
    
    print(f"Saving quantized model to {output_path}...")
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize model using AutoAWQ")
    parser.add_argument("--model_path", type=str, required=True, help="Path to merged HF model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save quantized model")
    args = parser.parse_args()
    
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM"
    }
    
    quantize(args.model_path, args.output_path, quant_config)
