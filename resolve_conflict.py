import sys

with open('plotting.ipynb', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
i = 0
conflict_idx = 0
while i < len(lines):
    if lines[i].startswith('<<<<<<< HEAD'):
        conflict_idx += 1
        head_lines = []
        i += 1
        while not lines[i].startswith('======='):
            head_lines.append(lines[i])
            i += 1
        i += 1 # skip =======
        remote_lines = []
        while not lines[i].startswith('>>>>>>>'):
            remote_lines.append(lines[i])
            i += 1
        i += 1 # skip >>>>>>>
        
        if conflict_idx == 1:
            new_lines.extend(head_lines)
        elif conflict_idx == 2:
            new_lines.extend(head_lines)
        elif conflict_idx == 3:
            new_lines.extend(head_lines)
            new_lines.extend(remote_lines[1:])
    else:
        new_lines.append(lines[i])
        i += 1

with open('plotting.ipynb', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Conflicts resolved:", conflict_idx)
