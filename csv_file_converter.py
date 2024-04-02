input_text = """
"""

# Prepare the input (split into lines and filter out unnecessary ones)
lines = input_text.strip().split('\n')
cleaned_lines = [line for line in lines if not all(char in '+-| ' for char in line)]

# Process each line to format as CSV
csv_lines = []
for line in cleaned_lines:
    parts = [part.strip() for part in line.split('|') if part]
    csv_line = ','.join(parts)
    csv_lines.append(csv_line)

# Combine all lines into a single CSV string
csv_content = "\n".join(csv_lines)

# Write to a CSV file
with open('/users/charles/desktop/pytorch-examples-tutorials/', 'w') as csv_file:
    csv_file.write(csv_content)

