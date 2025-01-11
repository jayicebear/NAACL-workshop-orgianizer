filename = '/home/jayicebear/snap/NAACL slang/result/0111_1742.txt'
f = open(filename, 'r')
slang = f.readlines()

unique_keys = set()
result = []
for line in slang:
    if line.strip():  # Skip empty lines
        if '\t' in line:  # Ensure the line contains a key-value pair
            key, value = line.split('\t', 1)
            if key not in unique_keys:
                unique_keys.add(key)
                result.append(line)
        else: 
            result.append(line)  # Preserve lines without '\t' (e.g., separators)
    else:
        result.append('\n') 

output_path = 'cleaned_unique_slang.txt'
with open(output_path, 'w', encoding='utf-8') as f:
    f.writelines(result)

output_path
