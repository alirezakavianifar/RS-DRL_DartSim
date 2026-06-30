import urllib.request
import json
import os
import re

def search_semantic_scholar():
    url = "https://api.semanticscholar.org/graph/v1/paper/search?query=DARTSim+An+Exemplar+for+Evaluation+and+Comparison&fields=title,authors,openAccessPdf,abstract,externalIds"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            print("Semantic Scholar Search Results:")
            for item in data.get('data', []):
                print(f"Title: {item.get('title')}")
                print(f"OpenAccess PDF: {item.get('openAccessPdf')}")
                print(f"Abstract: {item.get('abstract')[:200]}...")
                
                pdf_info = item.get('openAccessPdf')
                if pdf_info and pdf_info.get('url'):
                    pdf_url = pdf_info.get('url')
                    target_path = os.path.join("downloaded_articles", "SEAMS2019_DARTSim_Exemplar.pdf")
                    print(f"Downloading from {pdf_url} to {target_path}...")
                    req_pdf = urllib.request.Request(pdf_url, headers=headers)
                    with urllib.request.urlopen(req_pdf, timeout=30) as p_resp, open(target_path, 'wb') as out:
                        out.write(p_resp.read())
                    print("Download complete!")
                    
                    with open(os.path.join("downloaded_articles", "SEAMS2019_DARTSim_Exemplar_summary.txt"), "w", encoding="utf-8") as f:
                        f.write(f"Title: {item.get('title')}\nURL: {pdf_url}\n\nAbstract:\n{item.get('abstract')}\n")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    search_semantic_scholar()
