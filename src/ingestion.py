import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_document(file_path):
    """
    Charge un document (PDF ou Texte) et retourne son contenu brut.
    Détecte l'extension pour choisir le bon loader.
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        print(f"📄 Chargement du PDF : {file_path}")
        loader = PyPDFLoader(file_path)
        # PyPDFLoader extrait automatiquement le numéro de page dans les métadonnées
        return loader.load()
    
    elif ext in [".txt", ".md", ".py"]:
        print(f"📝 Chargement du texte : {file_path}")
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()
    
    else:
        raise ValueError(f"Format non supporté : {ext}")

def split_documents(documents):
    """
    Découpe les documents en morceaux (chunks) pour le RAG.
    On garde un overlap pour ne pas couper des phrases en plein milieu.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Taille de chaque morceau
        chunk_overlap=200,    # Chevauchement pour le contexte
        separators=["\n\n", "\n", " ", ""] # Priorité de découpage
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Découpage terminé : {len(chunks)} chunks créés.")
    return chunks

# --- Bloc de test rapide (pour vérifier si ça marche) ---
if __name__ == "__main__":
    # Pour tester, mets un fichier PDF bidon dans le dossier 'data'
    # et change le nom ici :
    test_file = "data/test.pdf" 
    
    if os.path.exists(test_file):
        docs = load_document(test_file)
        chunks = split_documents(docs)
        print(f"Exemple de chunk : {chunks[0].page_content[:100]}...")
        print(f"Source : {chunks[0].metadata}")
    else:
        print("⚠️  Aucun fichier de test trouvé dans data/. Ajoutes-en un pour tester !")