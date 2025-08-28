import json
import pandas as pd
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def process_nvd_dataset(input_path, output_dir="data"):
    """
    Process NVD CVE JSON data into structured format for vulnerability analysis
    """
    print(f"📂 Loading NVD data from: {input_path}")
    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Determine the correct key for CVE entries
    if 'CVE_Items' in data:
        items = data['CVE_Items']
    elif 'vulnerabilities' in data:
        items = [v['cve'] for v in data['vulnerabilities'] if 'cve' in v]
    else:
        raise KeyError("❌ JSON does not contain 'CVE_Items' or 'vulnerabilities'")

    cve_items = []

    for item in items:
        try:
            # NVD 2.0 format
            cve_id = item.get('id') or item.get('CVE_data_meta', {}).get('ID')
            if not cve_id:
                raise ValueError("Missing CVE ID")

            published_raw = item.get('published') or item.get('publishedDate')
            if isinstance(published_raw, dict):
                published_raw = published_raw.get('date')
            if not published_raw:
                raise ValueError("Missing published date")
            published_date = datetime.strptime(published_raw.split("T")[0], "%Y-%m-%d").date()

            # CVSS severity
            impact = item.get('metrics', {}) or item.get('impact', {})
            cvss_v3 = None
            if 'cvssMetricV31' in impact:
                cvss_v3 = impact['cvssMetricV31'][0].get('cvssData', {}).get('baseScore')
            elif 'cvssMetricV30' in impact:
                cvss_v3 = impact['cvssMetricV30'][0].get('cvssData', {}).get('baseScore')
            cvss_v2 = impact.get('baseMetricV2', {}).get('cvssV2', {}).get('baseScore')
            severity = cvss_v3 if cvss_v3 is not None else cvss_v2

            # Description
            desc_data = item.get('descriptions') or item.get('description', {}).get('description_data', [])
            description = desc_data[0]['value'] if desc_data else ""

            # CPE list
            cpe_list = []
            for node in item.get('configurations', {}).get('nodes', []):
                for cpe in node.get('cpe_match', []):
                    if 'cpe23Uri' in cpe:
                        cpe_list.append(cpe['cpe23Uri'])

            cve_items.append({
                'cve_id': cve_id,
                'published_date': published_date,
                'severity': severity,
                'description': description,
                'cpe_list': ' | '.join(cpe_list),
                'year': published_date.year
            })

        except Exception as e:
            print(f"⚠️ Skipped one CVE due to error: {e}")
            continue
        

    # Convert to DataFrame
    df = pd.DataFrame(cve_items)

    if df.empty or 'description' not in df.columns:
        print("❌ No valid CVE entries were processed. Exiting.")
        return

    # Vectorize descriptions
    print("🔠 Vectorizing vulnerability descriptions...")
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    description_vectors = vectorizer.fit_transform(df['description'])

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "nvd_processed.csv")
    df.to_csv(output_path, index=False)
    joblib.dump(vectorizer, os.path.join(output_dir, "nvd_vectorizer.pkl"))

    print(f"✅ NVD processing complete. Saved to: {output_path}")
    print(f"📊 Processed {len(df)} CVEs")

if __name__ == "__main__":
    input_file = "data/raw/network_logs/nvdcve-2.0-2025.json"
    process_nvd_dataset(input_file)
