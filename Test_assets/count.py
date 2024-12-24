import os
from collections import defaultdict

def count_image_formats(directory):
    image_formats = defaultdict(int)
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_extension = os.path.splitext(filename)[1].lower()
            if file_extension:
                image_formats[file_extension] += 1
    return image_formats

def update_description_file(description_file, image_formats):
    with open(description_file, 'a') as file:
        file.write("\n# Image Format Counts\n")
        for format, count in image_formats.items():
            file.write(f"- **{format}**: {count}\n")

if __name__ == "__main__":
    directory = r"D:\univlm\univlm\Test_assets"  # Update this path to your directory
    description_file = r"D:\univlm\univlm\Test_assets\description.md"  # Update this path to your description file

    image_formats = count_image_formats(directory)
    update_description_file(description_file, image_formats)