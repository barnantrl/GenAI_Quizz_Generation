import streamlit as st
import os
import tempfile
import json
from ingestion import load_document, split_documents
# On importe l'agent configuré dans l'étape précédente
from agent_logic import agent_executor 

# --- FONCTIONS UTILITAIRES (Le lien entre UI et Agent) ---

def create_vector_store(chunks):
    """
    Simule ou appelle la création du vector store.
    Note : Idéalement, cette fonction devrait être dans rag_chain.py 
    et mettre à jour la base de données vectorielle.
    """
    # Pour l'instant, on suppose que rag_chain gère ça ou on laisse passer
    # Si tu utilises ChromaDB en local, l'initialisation se fait souvent au chargement
    pass 

def generate_key_points(text):
    """Demande à l'agent de résumer les concepts clés."""
    prompt = f"""
    Analyse le texte suivant et identifie les 3 concepts clés principaux.
    Fais un résumé très concis sous forme de liste à puces.
    Texte : {text}
    """
    # On invoque l'agent
    response = agent_executor.invoke({"input": prompt})
    return response["output"]

def generate_quiz_question(topic):
    """
    Demande à l'agent de générer une question au format JSON strict
    pour que l'interface puisse l'afficher proprement.
    """
    prompt = f"""
    Agis comme un professeur expert. Génère une question de quiz (QCM) sur le sujet : "{topic}".
    
    IMPORTANT : Tu dois répondre UNIQUEMENT avec un objet JSON valide, sans texte avant ni après.
    Le format doit être exactement celui-ci :
    {{
        "question": "L'intitulé de la question ?",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "reponse_correcte": "Le texte exact de la bonne option",
        "explication": "Une explication pédagogique claire.",
        "citation_source": "Une citation courte du contexte qui prouve la réponse."
    }}
    """
    try:
        response = agent_executor.invoke({"input": prompt})
        # Nettoyage basique au cas où le LLM ajoute des ```json ... ```
        json_str = response["output"].replace("```json", "").replace("```", "").strip()
        return json.loads(json_str)
    except Exception as e:
        # En cas d'erreur de parsing JSON (ça arrive avec les LLM), on renvoie une erreur propre
        return {
            "question": "Erreur de génération",
            "options": ["Erreur"],
            "reponse_correcte": "Erreur",
            "explication": f"L'agent n'a pas renvoyé un JSON valide. Détail: {e}",
            "citation_source": "N/A"
        }

# --- DÉBUT DE L'APPLICATION STREAMLIT ---

st.set_page_config(page_title="GenAI Quiz - Glass Box", layout="wide")

st.title("🎓 Générateur de Quiz Pédagogique (Glass Box)")
st.markdown("""
Cette application transforme vos documents en quiz interactifs.
**Particularité :** Chaque réponse est justifiée par une preuve textuelle ("Glass Box").
""")

# --- SIDEBAR : INGESTION ---
with st.sidebar:
    st.header("1. Vos Données")
    uploaded_file = st.file_uploader("Déposez votre cours (PDF)", type=["pdf"])
    
    if uploaded_file:
        # On sauvegarde le fichier temporairement pour pouvoir le lire
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        if st.button("🚀 Analyser le document"):
            with st.spinner("Traitement du document & Vectorisation..."):
                # 1. Ingestion
                docs = load_document(tmp_path)
                chunks = split_documents(docs)
                
                # 2. Vectorisation (RAG)
                try:
                    create_vector_store(chunks)
                    st.success(f"Indexé ! ({len(chunks)} fragments)")
                except Exception as e:
                    st.warning(f"Mode sans Embedding (Vector Store non créé) : {e}")
                
                # 3. Extraction des points clés (Agent)
                # On prend juste le début du document pour le résumé global pour économiser des tokens
                summary = generate_key_points(docs[0].page_content[:2000])
                
                # On stocke tout en session pour ne pas perdre les données au clic
                st.session_state["summary"] = summary
                st.session_state["doc_processed"] = True
                
        # Nettoyage fichier temp
        # os.remove(tmp_path) # Commenté pour éviter les erreurs de permission windows parfois

# --- ZONE PRINCIPALE ---

if "doc_processed" in st.session_state:
    
    # SECTION 1 : Résumé des Concepts
    st.header("2. Concepts Clés Identifiés")
    st.info(st.session_state["summary"])
    
    st.divider()
    
    # SECTION 2 : Génération de Quiz
    st.header("3. Zone de Quiz")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        topic = st.text_input("Sujet de la question", "Le concept principal")
        if st.button("Générer une question"):
            with st.spinner("L'agent réfléchit..."):
                q_data = generate_quiz_question(topic)
                st.session_state["current_question"] = q_data
                st.session_state["user_answer"] = None # Reset réponse

    # Affichage de la question si elle existe
    if "current_question" in st.session_state:
        q = st.session_state["current_question"]
        
        with col2:
            st.subheader(f"❓ {q.get('question', 'Erreur')}")
            
            # Gestion du formulaire de réponse
            options = q.get("options", [])
            # Astuce : on utilise radio button
            choice = st.radio("Votre réponse :", options, key="radio_q")
            
            if st.button("Valider la réponse"):
                st.session_state["user_answer"] = choice

            # Feedback & Glass Box
            if st.session_state.get("user_answer"):
                is_correct = (st.session_state["user_answer"] == q["reponse_correcte"])
                
                if is_correct:
                    st.success("✅ Bonne réponse !")
                else:
                    st.error(f"❌ Incorrect. La bonne réponse était : {q['reponse_correcte']}")
                
                # --- LA GLASS BOX ---
                with st.expander("🔍 PREUVE (Glass Box) - Voir la source exacte", expanded=True):
                    st.markdown(f"**Explication de l'Agent :** {q['explication']}")
                    st.markdown("---")
                    st.markdown(f"**📜 Citation du document source :**")
                    st.caption(f"> \"{q['citation_source']}\"")
                    
                    meta = q.get("source_metadata", {})
                    if meta:
                        st.markdown(f"**📍 Localisation :** Page {meta.get('page', '?')}")

else:
    st.info("👈 Commencez par uploader un document dans la barre latérale.")