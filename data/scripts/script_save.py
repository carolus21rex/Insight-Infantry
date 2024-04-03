import os
import csv
import glob

def txt_to_csv(output_csv):
    # Open the CSV file for writing
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'content[1]', 'content[2]', 'content[1]+content[3]', 'content[2]+content[4]', 'content[0]'])

        # Iterate over all .txt files in the current working directory
        for txt_file in glob.glob('*.txt'):
            with open(txt_file, 'r') as file:
                # Read the contents of the file line by line
                for line in file:
                    contents = line.split()

                    # Check if the line has enough content to avoid IndexError
                    if len(contents) >= 5:
                        # Prepare the row data according to the specified format
                        row_data = [
                            os.path.basename(txt_file),
                            contents[1],
                            contents[2],
                            str(float(contents[1]) + float(contents[3])),
                            str(float(contents[2]) + float(contents[4])),
                            contents[0]
                        ]
                        # Write the row to the CSV file
                        writer.writerow(row_data)

# Example usage
output_csv = '0_output.csv'
txt_to_csv(output_csv)
