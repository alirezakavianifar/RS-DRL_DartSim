import os
import urllib.request
import json
import xml.etree.ElementTree as ET

def download_file(url, target_path, description):
    print(f"Downloading {description} from {url}...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response, open(target_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"Successfully saved to {target_path}")
        return True
    except Exception as e:
        print(f"Failed to download {description}: {e}")
        return False

def main():
    os.makedirs("downloaded_articles", exist_ok=True)
    
    # Direct papers and documentation links related to DARTSim and sCPS self-adaptation
    direct_papers = [
        {
            "url": "http://www.cs.cmu.edu/~able/publications/moreno-seams19.pdf",
            "filename": "Moreno_2019_DARTSim_An_Exemplar_for_Evaluation.pdf",
            "title": "DARTSim: An Exemplar for Evaluation and Comparison of Self-Adaptation Approaches for Smart Cyber-Physical Systems",
            "abstract": "DARTSim is an exemplar designed to enable researchers to evaluate and compare self-adaptation approaches for smart cyber-physical systems (sCPS). It simulates a team of unmanned aerial vehicles (UAVs) executing a reconnaissance mission in a hostile, unknown environment."
        },
        {
            "url": "https://arxiv.org/pdf/1905.03333.pdf",
            "filename": "ArXiv_1905.03333_DARTSim_Exemplar.pdf",
            "title": "DARTSim Exemplar Paper (ArXiv Version)",
            "abstract": "Self-adaptation approaches for smart cyber-physical systems."
        },
        {
            "url": "https://arxiv.org/pdf/2206.12492.pdf",
            "filename": "ArXiv_2206.12492_Self_Adaptation_Artifacts.pdf",
            "title": "Guidelines for Artifacts to Support Industry-Relevant Research on Self-Adaptation",
            "abstract": "Surveys and guidelines for self-adaptation exemplars including DARTSim."
        }
    ]
    
    for paper in direct_papers:
        target_pdf = os.path.join("downloaded_articles", paper["filename"])
        download_file(paper["url"], target_pdf, paper["title"])
        
        target_meta = os.path.join("downloaded_articles", paper["filename"].replace(".pdf", "_summary.txt"))
        with open(target_meta, "w", encoding="utf-8") as f:
            f.write(f"Title: {paper['title']}\nURL: {paper['url']}\n\nAbstract/Description:\n{paper['abstract']}\n")

    # Query arXiv API for additional related research papers
    queries = [
        'all:DARTSim',
        'all:"self-adaptation" AND all:sCPS',
        'all:"Gabriel Moreno" AND all:"self-adaptation"'
    ]
    
    seen_ids = set()
    for query in queries:
        url = f'http://export.arxiv.org/api/query?search_query={urllib.parse.quote(query)}&start=0&max_results=5'
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                xml_data = response.read().decode('utf-8')
            
            root = ET.fromstring(xml_data)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('atom:entry', ns):
                title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                paper_id = entry.find('atom:id', ns).text.strip().split('/')[-1]
                summary = entry.find('atom:summary', ns).text.strip()
                
                if paper_id in seen_ids:
                    continue
                seen_ids.add(paper_id)
                
                pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
                safe_title = "".join([c if c.isalnum() or c in " ._-" else "_" for c in title])[:50]
                pdf_path = os.path.join("downloaded_articles", f"ArXiv_{paper_id}_{safe_title}.pdf")
                txt_path = os.path.join("downloaded_articles", f"ArXiv_{paper_id}_{safe_title}_summary.txt")
                
                download_file(pdf_url, pdf_path, title)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(f"Title: {title}\nArXiv ID: {paper_id}\nURL: {pdf_url}\n\nAbstract:\n{summary}\n")
        except Exception as e:
            print(f"Error querying arXiv for {query}: {e}")

if __name__ == "__main__":
    main()
