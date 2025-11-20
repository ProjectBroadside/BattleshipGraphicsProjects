
import shutil
import sys
import json
import os

json_file_path = sys.argv[1]

with open(json_file_path, 'r') as f:
    file_pairs = json.load(f)

for source_path, destination_path in file_pairs:
    try:
        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.move(source_path, destination_path)
        print(f"Successfully moved {source_path} to {destination_path}")
    except FileNotFoundError:
        print(f"Error: Source file not found at {source_path}")
    except Exception as e:
        print(f"Error moving file {source_path}: {e}")
