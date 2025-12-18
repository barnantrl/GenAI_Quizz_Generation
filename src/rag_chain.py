import json
import re
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agent_logic import llm

def generate_key_points(docs):
    text_content = docs[0].page_content[:2000]
    prompt = PromptTemplate(
        template="""Tu es un expert pédagogique. Résume ce texte en 3 points clés factuels.
        TEXTE : {text}""",
        input_variables=["text"]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text_content})

def clean_json_string(json_str):
    try:
        start = json_str.find('{')
        end = json_str.rfind('}') + 1
        if start == -1 or end == 0: return ""
        json_clean = json_str[start:end]
        json_clean = re.sub(r',\s*}', '}', json_clean)
        json_clean = re.sub(r',\s*]', ']', json_clean)
        return json_clean
    except:
        return ""

def generate_quiz_question(topic, vector_store):
    print(f"RAG : Recherche de contexte pour '{topic}'...")
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(topic)
    
    if not docs:
        return fallback_response(topic, "Sujet non trouvé dans le document.")

    context = "\n\n".join([d.page_content for d in docs])
    meta = docs[0].metadata

    template = """
    Tu es un professeur expert en création d'examens.
    
    CONTEXTE PÉDAGOGIQUE :
    {context}
    
    TACHE : Créer une question QCM sur "{topic}".
    
    CONSIGNES DE RAISONNEMENT (CHAIN OF THOUGHT):
    Avant de générer la question, tu dois réfléchir étape par étape :
    1. ANALYSE : Identifie une phrase clé du texte qui contient une information vérifiable.
    2. FORMULATION : Crée une question basée uniquement sur cette phrase.
    3. DISTRACTEURS : Invente 3 mauvaises réponses plausibles mais clairement fausses d'après le texte.
    4. VÉRIFICATION : Est-ce que la réponse est explicitement dans le texte ? Si non, recommence.
    
    FORMAT DE SORTIE (JSON UNIQUEMENT) :
    {{
        "raisonnement_cot": "Etape 1: J'ai choisi la phrase... Etape 2: La question porte sur... Etape 3: J'ai vérifié que...",
        "question": "...",
        "options": ["...", "...", "...", "..."],
        "reponse_correcte": "...",
        "explication": "...",
        "citation_source": "La phrase exacte du texte utilisée."
    }}
    """
    
    prompt = PromptTemplate(template=template, input_variables=["topic", "context"])
    chain = prompt | llm | StrOutputParser()
    
    print(f"Raisonnement CoT en cours sur : {topic}...")
    raw_res = chain.invoke({"topic": topic, "context": context})

    try:
        json_clean = clean_json_string(raw_res)
        data = json.loads(json_clean)
        
        if "transformer" in topic.lower() and "megatron" in str(data).lower():
             return fallback_response(topic, "Hallucination détectée (Confusion avec le film).")

        data["source_metadata"] = meta
        return data
        
    except Exception as e:
        print(f"Erreur JSON/CoT : {e}")
        return fallback_response(topic, "Erreur de formatage du raisonnement.")

def fallback_response(topic, reason):
    return {
        "question": f"Impossible de générer le quiz sur '{topic}'",
        "options": ["Erreur", "Erreur", "Erreur", "Erreur"],
        "reponse_correcte": "Erreur",
        "explication": f"Échec du raisonnement : {reason}",
        "citation_source": "Système",
        "raisonnement_cot": "Le modèle n'a pas réussi à suivre la chaîne de pensée logique."
    }