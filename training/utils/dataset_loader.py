from datasets import load_dataset
import yaml

def format_instruction(sample):
    """
    Format the instruction into the Alpaca prompt template.
    Template:
    ### Instruction:
    {instruction}
    
    ### Input:
    {input}
    
    ### Response:
    {output}
    """
    instruction = sample['instruction']
    inp = sample.get('input', '')
    output = sample['output']
    
    if inp:
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
    else:
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    
    # Check if 'text' column exists, if not create it. Often needed by SFTTrainer
    sample['text'] = text
    return sample

def load_and_prepare_dataset(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    dataset_path = config['dataset_path']
    
    # Load dataset from JSONL
    # split='train' allows us to use standard dataset dict methods if needed, 
    # but here we might want to manually split if not already split
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    # Apply formatting
    processed_dataset = dataset.map(format_instruction)
    
    # Split into train/test
    # 90% train, 10% test
    split_dataset = processed_dataset.train_test_split(test_size=0.1)
    
    return split_dataset
