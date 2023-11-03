import os

def replace_first_element(file_path, desired_number):
    # Read the content of the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Modify each line by replacing the first element
    modified_lines = [f"{desired_number} {line.split(' ', 1)[1]}" for line in lines]

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.writelines(modified_lines)

def process_files_in_folder(folder_path, desired_number):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            replace_first_element(file_path, desired_number)

if __name__ == "__main__":
    folder_path = r"D:\talha\Item\new dataset\mobile\valid\labels"  # Replace this with the path to your folder
    desired_number = 0  # Replace this with the desired number

    process_files_in_folder(folder_path, desired_number)

