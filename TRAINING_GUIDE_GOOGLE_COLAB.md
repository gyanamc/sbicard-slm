# SBI Card SLM – Training on Google Colab (Step-by-Step)

This guide explains the project structure, data format, and how to train the **sbicard-slm** (Small Language Model) for SBI Card Q&A on **Google Colab**.

---

## 1. Project and model overview

### What this project does
- **Base model:** `mtgv/MobileLLaMA-1.4B-Chat` (1.4B parameters, chat-tuned).
- **Method:** QLoRA (4-bit quantized base model + LoRA adapters).
- **Task:** Instruction-following on **SBI Card** content: features, fees, rewards, etc.
- **Output:** Fine-tuned model saved as `sbicard-slm-1.4b` (adapters + tokenizer).

### Main folders
| Path | Purpose |
|------|--------|
| `config.yaml` | Model, dataset path, QLoRA/LoRA and training hyperparameters. |
| `training/` | `train_model.py` and `utils/dataset_loader.py` (loads JSONL, Alpaca formatting, 90/10 split). |
| `data_processing/` | Scripts to build and augment the instruction dataset; `processed_data/` holds JSONL files. |
| `quantization/` | Optional: convert and quantize for deployment (GGUF/AWQ). |
| `railway-deployment/` | Optional: deploy the model (e.g. FastAPI + LLaMA.cpp). |

---

## 2. Data understanding

### Required format (Alpaca-style instruction)
Each line in the training JSONL must have:

- **`instruction`** – User question (e.g. “What are the features of Titan SBI Card?”).
- **`input`** – Optional extra context (often `""`).
- **`output`** – Model answer (features, fees, rewards text, etc.).
- **`metadata`** (optional) – e.g. `type`: `"features"` / `"fees"` / `"rewards"`, `card_name`.

### How the training pipeline uses it
1. **Dataset loader** (`training/utils/dataset_loader.py`):
   - Reads `config['dataset_path']` from the YAML config.
   - Builds a single string per example in **Alpaca format**:
     - `### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}`
   - Stores it in a `text` field for SFTTrainer.
   - Splits 90% train / 10% test.

2. **Config dataset path:**  
   `dataset_path: "data_processing/processed_data/sbicard_instructions.jsonl"`  
   If this file is missing, `train_model.py` falls back to `data_processing/processed_data/test_augmented.jsonl` and writes a temporary `config_local.yaml` for the loader.

### Data you have in the repo
- **`data_processing/processed_data/test_instructions.jsonl`** – Base instruction set (one question per card/topic).
- **`data_processing/processed_data/test_augmented.jsonl`** – Same content plus paraphrased instructions (e.g. “key features” vs “features”) for more variety.
- **`sbicard_instructions.jsonl`** – Not in the repo; this is the “full” dataset path used in config. You either create it from raw data or upload it.

### Creating the full dataset (optional)
If you have **raw card data** (e.g. scraped JSONL with `card_name`, `features`, `fees_and_charges`, etc.):

```bash
# 1) Turn raw card JSONL → Alpaca instructions
python data_processing/create_instruction_pairs.py raw_cards.jsonl data_processing/processed_data/sbicard_instructions.jsonl

# 2) Augment (paraphrase questions) for better robustness
python data_processing/augment_data.py \
  data_processing/processed_data/sbicard_instructions.jsonl \
  data_processing/processed_data/sbicard_augmented.jsonl
```

Then in `config.yaml` set:
`dataset_path: "data_processing/processed_data/sbicard_augmented.jsonl"` (or keep `sbicard_instructions.jsonl` and point to the file you created).

---

## 3. Training configuration (config.yaml)

- **Model:** `mtgv/MobileLLaMA-1.4B-Chat`, output name `sbicard-slm-1.4b`.
- **QLoRA:** 4-bit NF4, LoRA r=16, alpha=32, dropout 0.05; target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- **Training:** 3 epochs, batch size 4 × gradient accumulation 4 → effective 16, max sequence length 512, LR 2e-4, cosine schedule, warmup 3%, fp16.
- **Checkpoints:** `./results`, save/eval every 100 steps, logging every 10.

For Colab (T4/V100/A100), you can leave this as-is or reduce `per_device_train_batch_size` to 2 and/or `num_train_epochs` to 1–2 if you hit OOM or want a quick run.

---

## 4. Step-by-step: Training on Google Colab

### Step 1: Open Colab and enable GPU
1. Go to [Google Colab](https://colab.research.google.com).
2. **Runtime → Change runtime type → Hardware accelerator: GPU** (T4 is enough; A100 if available).
3. Save.

### Step 2: Clone the repo and go to project root
In a cell:

```python
!git clone https://github.com/gyanamc/sbicard-slm.git
%cd sbicard-slm
```

(Or clone your own fork and `%cd sbicard-slm`.)

### Step 3: Install dependencies
```python
!pip install -r requirements.txt
```

If Colab already has some of these, that’s fine. Key packages: `torch`, `transformers`, `peft`, `bitsandbytes`, `accelerate`, `datasets`, `trl`, `pyyaml` (add `pyyaml` to requirements if missing).

### Step 4: (Optional) Use your own dataset
- **Option A – Use repo test data:**  
  Do nothing. The script will use `data_processing/processed_data/sbicard_instructions.jsonl` if present, else `test_augmented.jsonl`.

- **Option B – Upload your own JSONL:**  
  Upload a file (e.g. `sbicard_instructions.jsonl`) to `data_processing/processed_data/` in Colab (Files panel or `files.upload()`), and ensure `config.yaml` has:
  `dataset_path: "data_processing/processed_data/sbicard_instructions.jsonl"`.

- **Option C – Generate from raw data:**  
  Upload raw card JSONL, then run `create_instruction_pairs.py` and optionally `augment_data.py` as in section 2, then point `config.yaml` to the output file.

### Step 5: Run training
From the repo root (`sbicard-slm`):

```python
!python training/train_model.py
```

- Config is read from `config.yaml` in the current directory.
- If the configured dataset path is missing, the script switches to `test_augmented.jsonl` and uses a temporary config for the loader.
- Training writes checkpoints to `./results` and at the end saves the final model and tokenizer to `sbicard-slm-1.4b` in the current directory.

### Step 6: Save the trained model (Colab disk is temporary)
Before the runtime is recycled:

- **Download the folder:**
  ```python
  !zip -r sbicard-slm-1.4b.zip sbicard-slm-1.4b
  ```
  Then download `sbicard-slm-1.4b.zip` from the Colab file browser.

- **Or push to Hugging Face Hub:**  
  In `config.yaml` set `push_to_hub: true` and `hub_model_id: "your-username/sbicard-slm-1.4b"`, log in with `huggingface-cli login` or `notebook_login()` in a cell, then run training (or add a short script that loads the saved model and calls `push_to_hub()`).

---

## 5. Summary checklist

| Step | Action |
|------|--------|
| 1 | Colab: Runtime → GPU. |
| 2 | Clone repo, `%cd sbicard-slm`. |
| 3 | `pip install -r requirements.txt`. |
| 4 | Ensure dataset: use test data, or upload/create `sbicard_instructions.jsonl` (and set `config.yaml` if needed). |
| 5 | Run `python training/train_model.py`. |
| 6 | Download `sbicard-slm-1.4b` (zip) or push to Hub. |

---

## 6. Troubleshooting

- **CUDA out of memory:** Reduce `per_device_train_batch_size` to 2 (and optionally `max_seq_length` to 256 in config). Restart runtime and run again.
- **Dataset not found:** Confirm the file at `config['dataset_path']` exists under the repo root, or rely on the fallback to `test_augmented.jsonl`.
- **Config not found:** Run `train_model.py` from the repo root (`sbicard-slm`), not from `training/`.
- **Slow or no GPU:** In Colab, check Runtime → Change runtime type → GPU and that `torch.cuda.is_available()` is True in a cell.

Once training finishes, you can use the saved model with the inference wrapper in `railway-deployment/` or convert it with the scripts in `quantization/` for deployment.
