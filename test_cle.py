from openai import OpenAI
import os

# ⚠️ Colle ta NOUVELLE clé directement ici entre les guillemets pour tester
# (Supprime ce fichier après le test ou retire la clé !)
ma_cle = "sk-proj-TaNouvelleCleIci..."

try:
    client = OpenAI(api_key=ma_cle)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Dis 'Bonjour Barnabé' si ça marche !"}]
    )
    print("✅ SUCCÈS :", response.choices[0].message.content)
except Exception as e:
    print("❌ ERREUR :", e)