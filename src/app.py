import streamlit as st
import tempfile
import os
from ingestion import create_vector_store, load_document
from rag_chain import generate_key_points, generate_quiz_question

st.set_page_config(page_title="GenAI Quiz - Multimedia", layout="wide")

st.title("Générateur multimodal de quiz")
st.markdown("""
Architecture **RAG Multimodale** : Analysez des cours textuels ou des vidéos/audios.
Le système utilise **Whisper** pour transcrire les vidéos et **Mistral** pour le raisonnement.
""")

with st.sidebar:
    st.header("1. Document Source")
    uploaded_file = st.file_uploader("Fichier du cours", type=["pdf", "mp4"])
    
    if uploaded_file and st.button("Analyser"):
        with st.spinner("Traitement en cours (Transcription ou Lecture)..."):
            file_name = uploaded_file.name
            file_ext = os.path.splitext(file_name)[1].lower()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            try:
                vs = create_vector_store(tmp_path)
                
                if vs:
                    docs = load_document(tmp_path)
                    summary = generate_key_points(docs)
                    
                    st.session_state["vector_store"] = vs
                    st.session_state["summary"] = summary
                    st.session_state["processed"] = True
                    st.success("Analyse terminée !")
                else:
                    st.error("Erreur : Le fichier n'a pas pu être traité.")

            except Exception as e:
                st.error(f"Une erreur est survenue : {e}")
            
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

if st.session_state.get("processed"):
    
    with st.expander("Résumé du contenu", expanded=False):
        st.info(st.session_state['summary'])
    
    st.divider()
    
    st.header("2. Quiz & Raisonnement (CoT)")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        topic = st.text_input("Sujet du quiz")
        if st.button("Générer"):
            with st.spinner("L'agent réfléchit..."):
                q = generate_quiz_question(topic, st.session_state["vector_store"])
                st.session_state["curr_q"] = q
                st.session_state["user_ans"] = None

    if "curr_q" in st.session_state:
        q = st.session_state["curr_q"]
        
        with col2:
            with st.expander("Voir le Raisonnement (CoT)", expanded=True):
                st.write(q.get("raisonnement_cot", "Non disponible"))
            
            st.markdown("---")
            st.subheader(f"? {q.get('question', 'Erreur')}")
            
            opts = q.get("options", [])
            if len(opts) > 1:
                choice = st.radio("Votre réponse :", opts)
                if st.button("Valider"):
                    st.session_state["user_ans"] = choice
            
            if st.session_state.get("user_ans"):
                correct = q.get("reponse_correcte")
                if st.session_state["user_ans"] == correct:
                    st.success("Bravo !")
                else:
                    st.error(f"Faux. Réponse : {correct}")
                
                st.info(q.get("explication"))
                st.caption(f"Source : {q.get('citation_source')}")

else:
    st.info("Uploadez un PDF ou une vidéo MP4 pour commencer.")