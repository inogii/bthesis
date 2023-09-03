import fileinput

with fileinput.input(files=('caida_rels.txt'), inplace=True, encoding="utf-8") as f:
    for line in f:
        if not line.startswith('#'):
            values = line.strip().split('|')
            new_line = '|'.join(values[:-1])
            print(new_line)
        
with open('caida_rels.txt', 'r') as f:
    new_lines = set(f.readlines())

with open('asrel.txt', 'r') as f:
    existing_lines = set(f.readlines())

unique_lines = new_lines.union(existing_lines)

with open('asrel.txt', 'w') as f:
    f.writelines(unique_lines)