import os
import urllib.request
import json

def double_check_all():
    target_dir = os.path.join("downloaded_articles", "citations")
    os.makedirs(target_dir, exist_ok=True)
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    # Query CrossRef API for works mentioning DARTSim
    print("Querying CrossRef API...")
    crossref_url = "https://api.crossref.org/works?query=DARTSim&rows=30"
    try:
        req = urllib.request.Request(crossref_url, headers=headers)
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            items = data.get('message', {}).get('items', [])
            print(f"CrossRef returned {len(items)} items for query 'DARTSim'.")
            
            crossref_papers = []
            for item in items:
                title_list = item.get('title', [])
                title = title_list[0] if title_list else "Untitled"
                doi = item.get('DOI', '')
                year = item.get('issued', {}).get('date-parts', [[None]])[0][0]
                
                # Filter for software engineering / computer science works
                container = item.get('container-title', [])
                venue = container[0] if container else ""
                
                crossref_papers.append({
                    "title": title,
                    "doi": doi,
                    "year": year,
                    "venue": venue
                })
                
            print(f"Sample CrossRef titles:")
            for p in crossref_papers[:10]:
                print(f" - [{p['year']}] {p['title']} ({p['venue']})")
    except Exception as e:
        print(f"CrossRef Error: {e}")

if __name__ == "__main__":
    double_check_all()
