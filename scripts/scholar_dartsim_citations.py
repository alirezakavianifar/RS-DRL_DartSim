import os
import urllib.request
import json
import re

def search_citations():
    target_dir = os.path.join("downloaded_articles", "citations")
    os.makedirs(target_dir, exist_ok=True)
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'}
    
    # We will use OpenAlex API (free, open academic graph with full citation links and open access PDFs)
    # Search for works citing MorenoDART2019 or mentioning DARTSim
    print("Querying OpenAlex API for DARTSim citations and references...")
    openalex_url = "https://api.openalex.org/works?search=DARTSim&per_page=20"
    
    try:
        req = urllib.request.Request(openalex_url, headers=headers)
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            results = data.get('results', [])
            print(f"Found {len(results)} works in OpenAlex matching 'DARTSim'.")
            
            summary_records = []
            
            for idx, item in enumerate(results, start=1):
                title = item.get('title') or "Untitled"
                doi = item.get('doi') or ""
                publication_year = item.get('publication_year') or ""
                authors = [a.get('author', {}).get('display_name') for a in item.get('authorships', []) if a.get('author')]
                authors_str = ", ".join(authors[:3]) if authors else "Unknown Authors"
                
                oa_info = item.get('open_access') or {}
                oa_url = oa_info.get('oa_url') or ""
                
                abstract_inverted = item.get('abstract_inverted_index')
                abstract_text = ""
                if abstract_inverted:
                    word_positions = []
                    for word, positions in abstract_inverted.items():
                        for pos in positions:
                            word_positions.append((pos, word))
                    word_positions.sort()
                    abstract_text = " ".join([w for pos, w in word_positions])
                
                # Check how DARTSim is mentioned
                text_to_check = (title + " " + abstract_text).lower()
                is_case_study = False
                usage_type = "Mentioned / Cited"
                
                if "case study" in text_to_check or "exemplar" in text_to_check or "evaluated on dartsim" in text_to_check or "benchmark" in text_to_check:
                    is_case_study = True
                    usage_type = "Used as Case Study / Benchmark"
                elif "dartsim" in title.lower():
                    usage_type = "Introduced DARTSim Exemplar"
                    is_case_study = True
                
                safe_title = "".join([c if c.isalnum() or c in " ._-" else "_" for c in title])[:50].strip()
                record_file = os.path.join(target_dir, f"Paper_{idx}_{safe_title}.txt")
                
                with open(record_file, "w", encoding="utf-8") as f:
                    f.write(f"Title: {title}\n")
                    f.write(f"Authors: {authors_str}\n")
                    f.write(f"Year: {publication_year}\n")
                    f.write(f"DOI: {doi}\n")
                    f.write(f"Open Access URL: {oa_url}\n")
                    f.write(f"DARTSim Usage Classification: {usage_type}\n\n")
                    f.write(f"Abstract / Content Summary:\n{abstract_text}\n")
                
                # If there's an open access PDF URL, attempt to download
                pdf_downloaded = False
                if oa_url and oa_url.endswith(".pdf"):
                    pdf_path = os.path.join(target_dir, f"Paper_{idx}_{safe_title}.pdf")
                    try:
                        req_pdf = urllib.request.Request(oa_url, headers=headers)
                        with urllib.request.urlopen(req_pdf, timeout=15) as pdf_resp, open(pdf_path, 'wb') as pdf_out:
                            pdf_out.write(pdf_resp.read())
                        pdf_downloaded = True
                    except Exception as e:
                        pass
                
                summary_records.append({
                    "idx": idx,
                    "title": title,
                    "authors": authors_str,
                    "year": publication_year,
                    "usage": usage_type,
                    "pdf_downloaded": pdf_downloaded,
                    "record_file": record_file
                })
                print(f"[{idx}] {title} ({publication_year}) -> {usage_type}")
                
            return summary_records
    except Exception as e:
        print(f"Error searching OpenAlex: {e}")
        return []

if __name__ == "__main__":
    search_citations()
