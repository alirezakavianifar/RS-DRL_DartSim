import os

def clean():
    folder = "downloaded_articles"
    keep_prefixes = [
        "ArXiv_1905.03333_DARTSim_Exemplar",
        "ArXiv_2206.12492_Self_Adaptation_Artifacts",
        "Moreno_2019_DARTSim_An_Exemplar_for_Evaluation"
    ]
    
    for fname in os.listdir(folder):
        file_path = os.path.join(folder, fname)
        should_keep = any(fname.startswith(prefix) for prefix in keep_prefixes)
        if not should_keep:
            try:
                os.remove(file_path)
                print(f"Removed irrelevant/duplicate file: {fname}")
            except Exception as e:
                print(f"Error removing {fname}: {e}")

if __name__ == "__main__":
    clean()
