
# GPT + Pinecone API

Cette API permet de connecter un agent GPT personnalisé à un index Pinecone via FastAPI.

## 📁 Structure

- `main.py` : Point d'entrée de l'API
- `pinecone_utils.py` : Connexion à Pinecone
- `models.py` : Schéma de la requête
- `.env.example` : Exemple de configuration
- `render.yaml` : Configuration automatique pour déploiement Render

## 🚀 Déploiement Render

1. Crée un compte sur https://dashboard.render.com
2. Crée un nouveau Web Service
3. Upload ce projet ou connecte un dépôt GitHub
4. Renseigne les variables d’environnement à partir de `.env.example`
5. L’API sera accessible publiquement pour connexion à GPT

## 🔗 Endpoint

```
POST /search_vtc
{
  "question": "J'ai réussi l'examen théorique, que faire ?"
}
```

Réponse :
```json
{
  "answer": "Voici ce que vous devez faire après l’examen théorique..."
}
```
