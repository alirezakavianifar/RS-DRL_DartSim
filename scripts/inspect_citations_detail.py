import os
import glob

def inspect_details():
    target_dir = os.path.join("downloaded_articles", "citations")
    files = glob.glob(os.path.join(target_dir, "*.txt"))
    
    print(f"Inspecting {len(files)} citation summary files...\n")
    for f in sorted(files):
        with open(f, "r", encoding="utf-8") as file:
            content = file.read()
            lines = content.split("\n")
            title = lines[0] if len(lines) > 0 else ""
            year = lines[2] if len(lines) > 2 else ""
            usage = lines[5] if len(lines) > 5 else ""
            print(f"--- {os.path.basename(f)} ---")
            print(f"{title}")
            print(f"{year}")
            print(f"{usage}\n")

if __name__ == "__main__":
    inspect_details()
