import os
import glob

def inspect_case_study_texts():
    target_dir = os.path.join("downloaded_articles", "citations")
    papers_to_check = [
        "Paper_1_DARTSim_ An Exemplar for Evaluation and Comparison.txt",
        "Paper_8_Explaining quality attribute tradeoffs in automate.txt",
        "Paper_12_Information Reuse and Stochastic Search.txt",
        "Paper_14_Self-Adaptive Mechanisms for Misconfigurations in.txt",
        "Paper_15_CHESS_ A Framework for Evaluation of Self-Adaptive.txt",
        "Paper_17_Wildfire-UAVSim_ An Exemplar for Evaluation of Ada.txt"
    ]
    
    for fname in papers_to_check:
        fpath = os.path.join(target_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
                print(f"==================================================")
                print(f"FILE: {fname}")
                print(f"==================================================")
                print(content[:600])
                print("\n")

if __name__ == "__main__":
    inspect_case_study_texts()
