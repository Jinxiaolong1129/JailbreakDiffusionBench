import json

# Path to input file
input_file = 'data/harmful/diffusion_db/diffusion_db_harm_translate.json'
# Path to output file
output_file = 'data/harmful/diffusion_db/diffusion_db_harm_translate_2000.json'

# Read the JSON file
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Keep only the first 2000 prompts
if len(data['prompts']) > 2000:
    data['prompts'] = data['prompts'][:2000]

# Write the filtered data back to a new file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"Original prompt count: {len(json.load(open(input_file, 'r', encoding='utf-8'))['prompts'])}")
print(f"Filtered prompt count: {len(data['prompts'])}")
print(f"Filtered data saved to {output_file}")