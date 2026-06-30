import urllib.request
import re
import os

def search_cmu_able():
    # Search CMU ABLE group publication pages or web search links for PDF
    urls = [
        "http://www.cs.cmu.edu/~able/publications/moreno-seams19.pdf",
        "https://www.cs.cmu.edu/~gmoreno/publications/seams19-dartsim.pdf",
        "https://raw.githubusercontent.com/cps-sei/dartsim/master/docs/seams19-dartsim.pdf",
        "https://github.com/cps-sei/dartsim/raw/master/paper.pdf",
        "https://resources.sei.cmu.edu/asset_files/Presentation/2019_017_001_548685.pdf"
    ]
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    for url in urls:
        print(f"Trying {url}...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                content = resp.read()
                if len(content) > 10000 and content.startswith(b"%PDF"):
                    target = os.path.join("downloaded_articles", "SEAMS2019_DARTSim_Exemplar.pdf")
                    with open(target, "wb") as f:
                        f.write(content)
                    print(f"SUCCESS! Saved to {target} ({len(content)} bytes)")
                    return True
        except Exception as e:
            print(f"Failed {url}: {e}")

if __name__ == "__main__":
    search_cmu_able()
