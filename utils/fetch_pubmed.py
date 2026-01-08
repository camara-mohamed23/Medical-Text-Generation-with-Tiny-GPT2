import requests

def fetch_pubmed_summaries(keywords, retmax=50, output_file="data/medical_text.txt"):
    all_texts = []

    for term in keywords:
        print(f"🔹 Recherche PubMed pour '{term}'")
        # Étape 1 : récupérer les PMIDs
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": term,
            "retmax": retmax,
            "retmode": "json"
        }
        search_response = requests.get(search_url, params=search_params)
        pmids = search_response.json()["esearchresult"]["idlist"]

        if not pmids:
            continue

        # Étape 2 : récupérer les résumés
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "text"
        }
        fetch_response = requests.get(fetch_url, params=fetch_params)
        abstracts = fetch_response.text

        # Ajouter chaque résumé séparément
        abstracts = [a.strip() for a in abstracts.split("\n\n") if len(a.strip()) > 50]
        all_texts.extend(abstracts)
        print(f"✅ {len(abstracts)} résumés récupérés pour '{term}'")

    # Sauvegarder dans un fichier
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_texts))

    print(f"\n📄 Tous les résumés sauvegardés dans {output_file}")


if __name__ == "__main__":
    keywords = ["diabetes", "hypertension", "cancer", "asthma", "cardiovascular disease"]
    fetch_pubmed_summaries(keywords, retmax=50)
