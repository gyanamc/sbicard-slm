import json
import argparse
import sys
import os

def load_data(input_file):
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def generate_instructions(card_data):
    instructions = []
    card_name = card_data.get('card_name', 'SBI Credit Card')
    
    # 1. Features
    features = card_data.get('features', [])
    if features:
        feature_text = "\n".join([f"- {f}" for f in features])
        instructions.append({
            "instruction": f"What are the features of {card_name}?",
            "input": "",
            "output": f"The features of {card_name} include:\n{feature_text}",
            "metadata": {"type": "features", "card_name": card_name}
        })
        instructions.append({
            "instruction": f"Tell me about {card_name}.",
            "input": "",
            "output": f"{card_name} comes with the following key benefits:\n{feature_text}",
            "metadata": {"type": "overview", "card_name": card_name}
        })

    # 2. Fees
    fees = card_data.get('fees_and_charges', {}).get('raw_fee_text', [])
    if fees:
        fee_text = "\n".join([f"- {f}" for f in fees])
        instructions.append({
            "instruction": f"What is the annual fee for {card_name}?",
            "input": "",
            "output": f"The fee structure for {card_name} is as follows:\n{fee_text}",
            "metadata": {"type": "fees", "card_name": card_name}
        })
        instructions.append({
            "instruction": f"How much does {card_name} cost?",
            "input": "",
            "output": f"Here are the charges for {card_name}:\n{fee_text}",
            "metadata": {"type": "fees", "card_name": card_name}
        })

    # 3. Rewards (if separate from features, otherwise use generic question)
    # We can infer rewards from features if specific keywords exist
    reward_features = [f for f in features if 'Reward' in f or 'Points' in f or 'Cashback' in f]
    if reward_features:
        reward_text = "\n".join([f"- {f}" for f in reward_features])
        instructions.append({
            "instruction": f"What rewards does {card_name} offer?",
            "input": "",
            "output": f"{card_name} offers these rewards:\n{reward_text}",
            "metadata": {"type": "rewards", "card_name": card_name}
        })

    # Eligibility (placeholder as it wasn't fully extracted yet, but good to have structure)
    # eligibility = card_data.get('eligibility', [])
    # if eligibility: ...

    return instructions

def main():
    parser = argparse.ArgumentParser(description="Convert raw card data to Alpaca instruction format")
    parser.add_argument('input_file', help="Path to raw JSONL file")
    parser.add_argument('output_file', help="Path to output JSONL file")
    args = parser.parse_args()

    try:
        raw_data = load_data(args.input_file)
        print(f"Loaded {len(raw_data)} items from {args.input_file}")

        all_instructions = []
        for item in raw_data:
            all_instructions.extend(generate_instructions(item))

        print(f"Generated {len(all_instructions)} instruction pairs")

        with open(args.output_file, 'w') as f:
            for instr in all_instructions:
                f.write(json.dumps(instr) + '\n')
        
        print(f"Saved instructions to {args.output_file}")

    except Exception as e:
        print(f"Error processing data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
