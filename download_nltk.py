import nltk

def download_nltk_resources():
    """Download required NLTK resources for the application"""
    resources = [
        'punkt',
        'stopwords'
    ]
    
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            print(f"Successfully downloaded NLTK resource: {resource}")
        except Exception as e:
            print(f"Error downloading NLTK resource {resource}: {str(e)}")

if __name__ == "__main__":
    print("Downloading NLTK resources...")
    download_nltk_resources()
    print("Finished downloading NLTK resources.")
