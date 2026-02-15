import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

urls = [
    "https://www.sec.gov/Archives/edgar/data/356514/000119312525092333/d894729d485bpos.htm",
    "https://www.sec.gov/Archives/edgar/data/927751/000119312521146562/d176869d497.htm"
]

headers = {
    "User-Agent": "YourName your-email@example.com",  # SEC policy-compliant identity
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_contract_text(url):
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()  # 403/404 pe turant error dikhe

    soup = BeautifulSoup(response.text, "html.parser")
    elements = soup.find_all(["p", "div", "td"])

    clauses = []
    for element in elements:
        text = clean_text(element.get_text())
        if len(text) > 200:
            clauses.append(text)
    return clauses

all_clauses = []

for url in urls:
    print(f"Extracting from: {url}")
    extracted = extract_contract_text(url)
    all_clauses.extend(extracted)

print("Total extracted clauses:", len(all_clauses))

# Keep top 300 if too many
final_clauses = all_clauses[:300]

df = pd.DataFrame({
    "clause_text": final_clauses
})

df.to_csv("extracted1_real_contract.csv", index=False)

print("Saved clauses to extracted1_real_contract.csv")
