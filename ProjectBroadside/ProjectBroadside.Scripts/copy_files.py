
import shutil
import sys

source_path = sys.argv[1]
destination_path = sys.argv[2]

try:
    shutil.copy2(source_path, destination_path)
    print(f"Successfully copied {source_path} to {destination_path}")
except FileNotFoundError:
    print(f"Error: Source file not found at {source_path}")
except Exception as e:
    print(f"Error copying file: {e}")
