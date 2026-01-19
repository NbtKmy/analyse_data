import re
import json

count = 0
examples = []

with open('processed_data/train.jsonl') as f:
    for line in f:
        # Check for German quotation marks
        if '\u201e' in line or '\u201c' in line:
            count += 1
            if len(examples) < 10:
                data = json.loads(line)
                examples.append(data['input_text'])

print(f'Total lines with quoted articles: {count}')
print(f'\nExamples:')
for ex in examples:
    print(f'  {ex}')
    print()
