import json

count_special = 0
examples = []

with open('processed_data/train.jsonl', 'rb') as f:
    for i, line_bytes in enumerate(f):
        line = line_bytes.decode('utf-8')

        # Check for various special characters
        special_chars = []
        for char in line:
            code = ord(char)
            # Check for non-ASCII special characters in the text
            if code > 127 and code not in range(192, 255):  # Not just umlauts
                if char not in ['ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü', 'ß', 'é', 'è', 'à', 'ô']:
                    special_chars.append((char, code, hex(code)))

        if special_chars:
            count_special += 1
            if len(examples) < 5:
                data = json.loads(line)
                examples.append({
                    'line': i,
                    'text': data['input_text'][:100],
                    'special': special_chars[:5]
                })

print(f'Lines with special characters (non-umlaut): {count_special}')
print(f'\nFirst {len(examples)} examples:')
for ex in examples:
    print(f"\nLine {ex['line']}:")
    print(f"  Text: {ex['text']}")
    print(f"  Special chars: {ex['special']}")
