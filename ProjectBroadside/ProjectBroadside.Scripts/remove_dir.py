
import shutil
import sys

dir_to_remove = sys.argv[1]

try:
    shutil.rmtree(dir_to_remove)
    print(f"Successfully removed directory: {dir_to_remove}")
except FileNotFoundError:
    print(f"Error: Directory not found at {dir_to_remove}")
except Exception as e:
    print(f"Error removing directory: {e}")
