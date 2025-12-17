import time
import json

# --- CLASSE DE SIMULATION (MOCK) ---
class MockAgentExecutor:
    """
    Cette classe imite le comportement de LangChain mais sans IA.
    Elle renvoie des données 'en dur' pour tester l'interface.
    """
    
    def invoke(self, input_dict):
        """
        Simule la méthode .invoke() de l'agent.
        Analyse sommairement le texte pour savoir si on veut un résumé ou un quiz.
        """
        user_input = input_dict.get("input", "").lower()
        
        # Simulation d'un petit temps de réflexion (pour l'effet UI)
        time.sleep(1.5)

        # CAS 1 : Demande de génération de QUIZ (détecté via mots clés)
        if "quiz" in user_input or "json" in user_input:
            # On renvoie un JSON simulé strict, comme le ferait l'IA
            fake_json = {
                "question": "Ceci est une question simulée (Mode Façade) ?",
                "options": [
                    "Réponse A (Fausse)", 
                    "Réponse B (Correcte)", 
                    "Réponse C (Fausse)", 
                    "Réponse D (Fausse)"
                ],
                "reponse_correcte": "Réponse B (Correcte)",
                "explication": "Ceci est une explication codée en dur pour tester l'affichage 'Glass Box' sans consommer d'API.",
                "citation_source": "Extrait simulé du document PDF, page 12."
            }
            return {"output": json.dumps(fake_json)}

        # CAS 2 : Demande de RÉSUMÉ (par défaut)
        else:
            return {"output": """
            • Concept Clé 1 (Simulé) : Le mode façade permet de tester l'UI.
            • Concept Clé 2 (Simulé) : Il n'y a pas d'appel API réel ici.
            • Concept Clé 3 (Simulé) : Tout fonctionne 'pour de faux'.
            """}

# --- CRÉATION DE L'AGENT ---
# Au lieu de créer un vrai agent LangChain, on instancie notre simulateur.
# app.py ne verra pas la différence car il appelle aussi .invoke()
agent_executor = MockAgentExecutor()