import json
import sys

def validate_scraper_output(file_path):
    print(f"--- Validating Scraper Output: {file_path} ---")
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        print(f"Total items scraped: {len(lines)}")
        
        for i, line in enumerate(lines[:3]): # Check first 3 items
            item = json.loads(line)
            print(f"\nItem {i+1}:")
            print(f"  Card Name: {item.get('card_name', 'N/A')}")
            print(f"  URL: {item.get('url', 'N/A')}")
            
            features = item.get('features', [])
            print(f"  Features Count: {len(features)}")
            if features:
                print(f"  First Feature: {features[0][:50]}...")
                
            fees = item.get('fees_and_charges', {})
            print(f"  Fees Data: {fees}")
            
    except Exception as e:
        print(f"Error reading file: {e}")

def validate_instructions(file_path):
    print(f"\n--- Validating Instructions: {file_path} ---")
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        print(f"Total instructions: {len(lines)}")
        
        valid_count = 0
        for i, line in enumerate(lines):
            item = json.loads(line)
            if all(k in item for k in ('instruction', 'input', 'output')):
                valid_count += 1
            if i < 3:
                print(f"\nInstruction {i+1}:")
                print(f"  Q: {item['instruction']}")
                print(f"  A: {item['output'][:100]}...")
        
        print(f"\nValid Alpaca format items: {valid_count}/{len(lines)}")
            
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        mode = sys.argv[1]
        file_path = sys.argv[2]
        
        if mode == 'scraper':
            validate_scraper_output(file_path)
        elif mode == 'instructions':
            validate_instructions(file_path)
        else:
            print("Unknown mode. Use 'scraper' or 'instructions'")
    elif len(sys.argv) > 1:
         # Default to scraper for backward compatibility or infer
         validate_scraper_output(sys.argv[1])
    else:
        print("Usage: python validate_dataset.py <mode: scraper|instructions> <file_path>")
