# import os

# def rename_images(directory):
#     for filename in os.listdir(directory):
#         if filename.startswith("demo") and filename.endswith(".jpg"):
#             new_name = filename.replace("demo", "input", 1)
#             os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))
#             print(f'Renamed: {filename} to {new_name}')

#         elif filename.startswith("demo") and filename.endswith(".jpeg"):
#             new_name = filename.replace("demo", "input", 1)
#             os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))
#             print(f'Renamed: {filename} to {new_name}')
        
#         elif filename.startswith("demo") and filename.endswith(".JPG"):
#             new_name = filename.replace("demo", "input", 1)
#             os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))
#             print(f'Renamed: {filename} to {new_name}')
        
#         elif filename.startswith("demo") and filename.endswith(".png"):
#             new_name = filename.replace("demo", "input", 1)
#             os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))
#             print(f'Renamed: {filename} to {new_name}')

#         else:
#             print(f'File {filename} does not match the pattern')
        
# if __name__ == "__main__":
#     directory = r"D:\univlm\univlm\Test_assets"  # Update this path to your directory
#     rename_images(directory)



# import os

# def rename_images(directory):
#     for filename in os.listdir(directory):
#         file_path = os.path.join(directory, filename)
#         if os.path.isfile(file_path):
#             file_extension = os.path.splitext(filename)[1]
#             new_name = f"input{file_extension}"
#             new_path = os.path.join(directory, new_name)
#             os.rename(file_path, new_path)
#             print(f'Renamed: {filename} to {new_name}')

# if __name__ == "__main__":
#     directory = r"D:\univlm\univlm\Test_assets"  # Update this path to your directory
#     rename_images(directory)


import os

def rename_images(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_extension = os.path.splitext(filename)[1]
            if file_extension in ['.py', '.md']:
                continue
            base_name = "input"
            new_name = f"{base_name}{file_extension}"
            new_path = os.path.join(directory, new_name)
            counter = 1
            while os.path.exists(new_path):
                new_name = f"{base_name}{counter:02d}{file_extension}"
                new_path = os.path.join(directory, new_name)
                counter += 1
            os.rename(file_path, new_path)
            print(f'Renamed: {filename} to {new_name}')

if __name__ == "__main__":
    directory = r"D:\univlm\univlm\Test_assets"  # Update this path to your directory
    rename_images(directory)

# This script will:

# Iterate through all files in the specified directory.
# Extract the file extension.
# Skip renaming if the file extension is .py or .md.
# Rename the file to input followed by its original extension.
# If a file with the new name already exists, it will append a number to the base name and increment it until a unique name is found.
# Print the renaming operation.