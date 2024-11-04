import os
import shutil
import random

# Dictionary mapping source folders to destination folders
folder_mapping = {
    # Numbers
    '0B': '0', '0G': '0', '0R': '0', '0Y': '0',
    '1B': '1', '1G': '1', '1R': '1', '1Y': '1',
    '2B': '2', '2G': '2', '2R': '2', '2Y': '2',
    '3B': '3', '3G': '3', '3R': '3', '3Y': '3',
    '4B': '4', '4G': '4', '4R': '4', '4Y': '4',
    '5B': '5', '5G': '5', '5R': '5', '5Y': '5',
    '6B': '6', '6G': '6', '6R': '6', '6Y': '6',
    '7B': '7', '7G': '7', '7R': '7', '7Y': '7',
    '8B': '8', '8G': '8', '8R': '8', '8Y': '8',
    '9B': '9', '9G': '9', '9R': '9', '9Y': '9',
    # Action cards
    'P2B': 'DRAW2', 'P2G': 'DRAW2', 'P2R': 'DRAW2', 'P2Y': 'DRAW2',
    'RVB': 'REVERSE', 'RVG': 'REVERSE', 'RVR': 'REVERSE', 'RVY': 'REVERSE',
    'SKPB': 'SKIP', 'SKPG': 'SKIP', 'SKPR': 'SKIP', 'SKPY': 'SKIP',
    # Wild cards
    'CLR': 'WILD',
    'PLUS4': 'DRAW4'
}

def get_unique_filename(source_folder, original_filename):
    """Create unique filename by prepending source folder name."""
    prefix = source_folder
    return f"{prefix}_{original_filename}"

def organize_cards(source_base_dir, dest_base_dir):
    # Create destination folders if they don't exist
    for dest_folder in set(folder_mapping.values()):
        dest_path = os.path.join(dest_base_dir, dest_folder)
        os.makedirs(dest_path, exist_ok=True)

    # Process each source folder
    for source_folder, dest_folder in folder_mapping.items():
        source_path = os.path.join(source_base_dir, source_folder)
        
        if not os.path.exists(source_path):
            print(f"Warning: Source folder '{source_folder}' not found.")
            continue

        # Get all valid image files, sorted alphabetically, excluding Zone Identifier files
        images = sorted([
            f for f in os.listdir(source_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and 'zone.identifier' not in f.lower()
        ])

        # Select every third image starting from the first (1 in 3)
        selected_images = images[0::3]

        for selected_image in selected_images:
            # Create unique filename
            unique_filename = get_unique_filename(source_folder, selected_image)
            
            source_file = os.path.join(source_path, selected_image)
            dest_file = os.path.join(dest_base_dir, dest_folder, unique_filename)
            
            try:
                # Copy the file with the unique filename
                shutil.copy2(source_file, dest_file)
                print(f"Copied '{selected_image}' from '{source_folder}' to '{dest_folder}' as '{unique_filename}'.")
            except Exception as e:
                print(f"Failed to copy '{selected_image}' from '{source_folder}': {e}")

if __name__ == "__main__":
    SOURCE_DIR = r"./data/data_pool"
    DEST_DIR = r"./data/new_pool"
    
    organize_cards(SOURCE_DIR, DEST_DIR)