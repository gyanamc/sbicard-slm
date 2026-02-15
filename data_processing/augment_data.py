import json
import argparse
import random

SYNONYMS = {
    "features": ["benefits", "attributes", "characteristics", "key features", "highlights"],
    "cost": ["price", "charge", "fee", "expense"],
    "annual fee": ["yearly charge", "annual cost", "membership fee", "renewal fee"],
    "rewards": ["reward points", "loyalty benefits", "bonuses", "cashback"],
    "eligibility": ["requirements", "who can apply", "prerequisites", "qualification criteria"]
}

TEMPLATES = {
    "features": [
        "What are the {synonym} of {card_name}?",
        "List the {synonym} for {card_name}.",
        "Can you tell me about the {synonym} of {card_name}?",
        "Describe the {synonym} included with {card_name}."
    ],
    "fees": [
        "What is the {synonym} for {card_name}?",
        "How much is the {synonym} of {card_name}?",
        "Tell me the {synonym} structure for {card_name}."
    ],
    "rewards": [
        "What {synonym} can I earn with {card_name}?",
        "Does {card_name} offer any {synonym}?",
        "Explain the {synonym} program for {card_name}."
    ]
}

def augment_instruction(item):
    augmented_items = []
    
    # Original item
    augmented_items.append(item)
    
    card_name = item['metadata'].get('card_name', 'this card')
    instr_type = item['metadata'].get('type', '')
    
    if instr_type in TEMPLATES:
        templates = TEMPLATES[instr_type]
        synonyms = SYNONYMS.get(instr_type, [instr_type])
        
        # Generate 2 variations
        for _ in range(2):
            template = random.choice(templates)
            synonym = random.choice(synonyms)
            new_instruction = template.format(synonym=synonym, card_name=card_name)
            
            if new_instruction != item['instruction']:
                new_item = item.copy()
                new_item['instruction'] = new_instruction
                augmented_items.append(new_item)
                
    return augmented_items

def main():
    parser = argparse.ArgumentParser(description="Augment instruction dataset")
    parser.add_argument('input_file', help="Path to input JSONL instructions")
    parser.add_argument('output_file', help="Path to output augmented JSONL")
    args = parser.parse_args()

    try:
        with open(args.input_file, 'r') as f:
            lines = f.readlines()
            
        print(f"Loaded {len(lines)} original instructions")
        
        all_augmented = []
        for line in lines:
            item = json.loads(line)
            augmented = augment_instruction(item)
            all_augmented.extend(augmented)
            
        print(f"Generated {len(all_augmented)} total instructions after augmentation")
        
        with open(args.output_file, 'w') as f:
            for item in all_augmented:
                f.write(json.dumps(item) + '\n')
                
        print(f"Saved to {args.output_file}")

    except Exception as e:
        print(f"Error during augmentation: {e}")

if __name__ == "__main__":
    main()
