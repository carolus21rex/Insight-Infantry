import os

# Path to the folder containing the txt files
folder_path = os.getcwd()

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    # Check if the file is a txt file
    if filename.endswith('.txt'):
        # Open the file and read its contents
        with open(os.path.join(folder_path, filename), 'r') as file:
            # Iterate through each line in the file
            for line in file:
                # Print the portion of the line up to the first space
                data = line.split(' ')
                expected_length = 4
                try:
                    if int(data[0]) not in (15, 16):
                        print(f"file: {filename} ; value:{data[0]}")
                    for split_ in range(1, expected_length):
                        if float(data[split_]) < 0 or float(data[split_]) > 1:
                            print(f"file: {filename} ; value:{data[0]}")
                except ValueError:
                    print(f"file: {filename} ; value:{data[0]}")
