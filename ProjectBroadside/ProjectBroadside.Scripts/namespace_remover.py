import os
import re

def remove_namespaces(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".cs"):
                filepath = os.path.join(root, file)
                content = None
                # Try to read with different encodings
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    try:
                        with open(filepath, 'r', encoding='latin-1') as f:
                            content = f.read()
                    except Exception as e:
                        print(f"Could not read file {filepath}: {e}")
                        continue
                
                if content is None:
                    continue

                # This regex is designed to handle various namespace formats, including nested ones.
                content = re.sub(r'namespace\s+[\w\.]+\s*\{\s*', '', content)
                # Remove the corresponding closing brace
                content = content.rstrip()
                if content.endswith('}'):
                    content = content[:-1].rstrip()

                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)

if __name__ == "__main__":
    # The script is in .Staging, so the directory is "."
    remove_namespaces(".")
    print("Namespace removal complete.")
