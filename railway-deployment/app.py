from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_wrapper import LlamaWrapper
import os
import time

app = FastAPI(title="SBI Card SLM API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "models/sbicard-slm.gguf")

class ChatRequest(BaseModel):
    query: str
    max_tokens: int = 256
    temperature: float = 0.7

@app.on_event("startup")
def startup_event():
    global model
    if os.path.exists(MODEL_PATH):
        model = LlamaWrapper(model_path=MODEL_PATH)
    else:
        print(f"WARNING: Model not found at {MODEL_PATH}. Waiting for file...")
        # In Docker, we might need to wait for volume mount or copy
        # For now, we fail if model is missing at startup or handle it gracefully
        pass

@app.get("/health")
def health_check():
    if model:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy", "model_loaded": False, "detail": "Model not loaded"}

@app.post("/v1/chat")
async def chat(request: ChatRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    start_time = time.time()
    
    # Format prompt for Alpaca
    formatted_prompt = f"### Instruction:\n{request.query}\n\n### Response:\n"
    
    try:
        response = model.generate(
            prompt=formatted_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        generated_text = response['choices'][0]['text'].strip()
        usage = response['usage']
        
        process_time = time.time() - start_time
        
        return {
            "response": generated_text,
            "usage": usage,
            "process_time": f"{process_time:.2f}s"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
