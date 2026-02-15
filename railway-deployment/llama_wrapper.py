from llama_cpp import Llama
import os

class LlamaWrapper:
    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int = None):
        """
        Initialize the Llama model.
        
        Args:
            model_path: Path to the GGUF model file.
            n_ctx: Context window size (default 2048).
            n_threads: Number of threads to use. Defaults to 4 or CPU count.
        """
        if n_threads is None:
            # Optimal for Railway usually relates to vCPUs allocated
            n_threads = max(1, os.cpu_count() - 1)
            
        print(f"Loading model from {model_path} with {n_threads} threads...")
        
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            # n_gpu_layers=0, # CPU only for Railway usually
            verbose=True
        )
        print("Model loaded successfully.")

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7, stop: list = None):
        """
        Generate text completion.
        """
        if stop is None:
            stop = ["### Instruction", "### User"]

        # Format prompt nicely if raw text is passed, though ideally the caller formats it
        # We assume the caller passes a formatted prompt (Alpaca style)
        
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            echo=False
        )
        
        return output
