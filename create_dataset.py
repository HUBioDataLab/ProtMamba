import csv

# Define the input and output file paths
input_file = "/media/ubuntu/8TB/mennan/data/uniref50.fasta"
output_file = 'protein_sequences.csv'

# Open the input file and read the lines
with open(input_file, 'r') as txt_file:
    lines = txt_file.readlines()

# Open the output file and write the sequences to it
with open(output_file, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    
    # Write the header
    writer.writerow(['sequence'])
    
    # Write each line from the txt file as a new row in the csv file
    for line in lines:
        # Strip any leading/trailing whitespace characters
        sequence = line.strip()
        # Write the sequence to the csv file
        writer.writerow([sequence])

print(f"CSV file '{output_file}' created successfully.")