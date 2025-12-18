# GenAI Project : Générateur de Quiz Multimodal avec Raisonnement

Ce projet est un assistant pédagogique intelligent capable de générer des quiz et des résumés à partir de documents de cours (PDF) ou de vidéos (MP4). Il utilise une architecture RAG locale pour garantir que les réponses sont strictement basées sur le contenu fourni.

## Fonctionnalités
- **Support Vidéo & Texte** : Transcrit les vidéos (Whisper) et lit les PDF pour créer une base de connaissances unifiée.
- **RAG Local** : Utilise `Mistral` et `nomic-embed-text` via Ollama pour une confidentialité totale des données.
- **Transparence (Glass Box)** : Affiche le raisonnement de l'IA avant la génération de la question.

## Choix de la Technique de Raisonnement : Chain of Thought (CoT)

Pour répondre aux exigences de fiabilité et de transparence pédagogique, nous avons choisi d'implémenter la technique du **Chain of Thought (CoT)** plutôt qu'une simple approche "Direct Prompting" ou "ReAct".

### Pourquoi ce choix ?
La génération de questions d'examen (QCM) demande une rigueur logique stricte pour éviter deux problèmes majeurs des LLM :
1. **Les Hallucinations :** Inventer des faits absents du cours (ex: parler du film *Transformers* au lieu du modèle NLP).
2. **Les erreurs de logique :** Proposer une bonne réponse qui est en fait fausse, ou des mauvaises réponses trop évidentes.

### Implémentation dans le projet
Nous avons structuré le prompt du modèle pour forcer une réflexion séquentielle en 4 étapes avant la production du JSON final :

1.  **ANALYSE :** L'agent doit d'abord identifier une phrase factuelle précise dans le contexte récupéré.
2.  **FORMULATION :** Il rédige la question basée uniquement sur cette preuve.
3.  **DISTRACTEURS :** Il génère activement des mauvaises réponses plausibles en se basant sur le contexte (pour éviter les options absurdes).
4.  **VÉRIFICATION :** L'agent effectue une auto-critique pour confirmer que la réponse est bien présente dans le texte source.

Cette chaîne de pensée est ensuite extraite et affichée à l'utilisateur dans l'interface (section "Glass Box"), garantissant que l'outil n'est pas une simple "boîte noire".

## Installation et Lancement

### Prérequis
- Python 3.10+
- [Ollama](https://ollama.com/) installé et lancé.
- Modèles Ollama : `ollama pull mistral` et `ollama pull nomic-embed-text`.

### Installation
1. Cloner le repo :
   ```bash
   git clone <votre-lien-repo>