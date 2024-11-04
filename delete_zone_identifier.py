import os

def delete_zone_identifier_files(parent_directory):
    """
    Delete all files containing 'zone.identifier' in their name within each subfolder of the specified parent directory.
    """
    for folder in os.listdir(parent_directory):
        folder_path = os.path.join(parent_directory, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if 'zone.identifier' in file.lower():
                    file_path = os.path.join(folder_path, file)
                    try:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    except Exception as e:
                        print(f"Failed to delete {file_path}: {e}")

if __name__ == "__main__":
    # Replace with your target directory
    TARGET_DIR = r"./data/data_pool"
    delete_zone_identifier_files(TARGET_DIR)