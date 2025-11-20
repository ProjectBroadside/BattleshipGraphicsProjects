
import shutil
import sys
import os

source_path = sys.argv[1]
destination_path = sys.argv[2]

try:
    # Ensure the destination directory exists
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.move(source_path, destination_path)
    print(f"Successfully moved {source_path} to {destination_path}")
except FileNotFoundError:
    print(f"Error: Source file not found at {source_path}")
except Exception as e:
    print(f"Error moving file: {e}")
