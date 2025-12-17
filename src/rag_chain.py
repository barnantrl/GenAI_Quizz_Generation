import os
import shutil
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# On charge les variables d'environnement (clé API)
load_dotenv()

# Dossier où sera stockée la base de données vectorielle (sur ton disque)
PERSIST_DIRECTORY = "./chroma_db"

def create_vector_store(chunks):
    """
    Prend les chunks de texte, calcule leurs vecteurs (embeddings)
    et les stocke dans ChromaDB.
    """
    # 1. On nettoie l'ancienne base pour éviter les doublons si on relance
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)

    # 2. Initialisation du modèle d'embedding (OpenAI)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print("💾 Création de la base vectorielle (Vector Store)...")
    
    # 3. Création et persistance automatique
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    
    print(f"✅ Base vectorielle créée avec {len(chunks)} fragments !")
    return vector_store.as_retriever()

def get_retriever():
    """
    Charge la base vectorielle existante pour faire des recherches.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    
    # On configure le retriever pour qu'il renvoie les 3 morceaux les plus pertinents
    return vector_store.as_retriever(search_kwargs={"k": 3})

# --- Bloc de test rapide ---
if __name__ == "__main__":
    # Pour tester ce fichier, on a besoin de chunks. 
    # On va réutiliser ingestion.py pour en générer vite fait.
    from ingestion import load_document, split_documents
    
    test_file = "data/test.pdf" # Assure-toi que ce fichier existe
    if os.path.exists(test_file):
        # 1. Ingestion
        docs = load_document(test_file)
        chunks = split_documents(docs)
        
        # 2. Vectorisation (Sauvegarde)
        retriever = create_vector_store(chunks)
        
        # 3. Test de récupération (Retrieval)
        question = "De quoi parle ce document ?" 
        results = retriever.invoke(question)
        
        print(f"\n❓ Question test : {question}")
        print(f"🔎 Résultat trouvé ({len(results)} chunks) :")
        print(f"   --- Extrait : {results[0].page_content[:150]}...")
        print(f"   --- Source : {results[0].metadata}")
    else:
        print("⚠️ Fichier test non trouvé.")