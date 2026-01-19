import json
import sys

def clean_text(text):
    """Remove special quote markers and clean text"""
    # Replace Windows-1252 smart quotes that got misencoded
    # \x98 and \x9c are start/end quote markers
    text = text.replace('\x98', '')  # Remove start quote marker
    text = text.replace('\x9c', '')  # Remove end quote marker

    # Also handle other common quote marks
    text = text.replace('\u201e', '')  # „
    text = text.replace('\u201c', '')  # "
    text = text.replace('\u201d', '')  # "
    text = text.replace('\u2018', '')  # '
    text = text.replace('\u2019', '')  # '

    # Clean up any double spaces
    while '  ' in text:
        text = text.replace('  ', ' ')

    return text.strip()

def process_file(input_file, output_file):
    """Process a JSONL file and clean special characters"""
    cleaned_count = 0
    total_count = 0

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:

        for line in fin:
            total_count += 1
            data = json.loads(line)

            original_text = data['input_text']
            cleaned_text = clean_text(original_text)

            if original_text != cleaned_text:
                cleaned_count += 1

            data['input_text'] = cleaned_text
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')

    return cleaned_count, total_count

# Process all three files
for dataset in ['train', 'val', 'test']:
    input_file = f'processed_data/{dataset}.jsonl'
    output_file = f'processed_data/{dataset}_cleaned.jsonl'

    print(f'Processing {dataset}.jsonl...')
    cleaned, total = process_file(input_file, output_file)
    print(f'  Cleaned {cleaned}/{total} lines ({100*cleaned/total:.1f}%)')
    print()

print('Done! Cleaned files saved with _cleaned suffix.')
print('Review the cleaned files and then rename them if satisfied.')
