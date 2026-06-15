import json

with open('explore_data_csv.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    cell_type = c['cell_type']
    source = ''.join(c['source'])
    print(f"=== Cell {i} ({cell_type}) ===")
    print(source)
    print()
