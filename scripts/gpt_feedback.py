import os
from openai import OpenAI

# Configuration du client OpenAI
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def generate_feedback(activity_type, observations, metrics=None):
    """
    Génère un feedback personnalisé basé sur l'analyse biomécanique
    
    Args:
        activity_type (str): Type d'activité analysée
        observations (str): Observations techniques de l'analyse
        metrics (dict): Métriques détaillées (optionnel)
    
    Returns:
        str: Feedback formaté pour l'utilisateur
    """
    
    # Ajouter les métriques au prompt si disponibles
    metrics_info = ""
    if metrics:
        metrics_info = f"""
        
**Métriques quantitatives détectées :**
- Score global : {metrics.get('overall_score', 0):.1f}/100
- Équilibre moyen : {metrics.get('avg_balance', 0):.1f}/100
- Symétrie corporelle : {metrics.get('avg_symmetry', 0):.1f}/100
- Posture moyenne : {metrics.get('avg_posture', 0):.1f}/100
- Stabilité : {metrics.get('balance_stability', 0):.1f}/100
        """
    
    prompt = f"""
Tu es un expert en biomécanique humaine, en kinésiologie appliquée, et en coaching personnalisé de haut niveau. 
Tu travailles avec des athlètes, des seniors, des personnes en rééducation, et des professionnels du mouvement.

Tu viens d'observer une séquence vidéo dans laquelle une personne réalise un mouvement de type : **{activity_type}**.

Les systèmes d'analyse IA de posture et de mouvement ont automatiquement généré les **observations suivantes** : 

{observations}

{metrics_info}

---

Ta mission :

🔍 Analyse les observations comme un professionnel : postures, alignement articulaire, fluidité, coordination, effort musculaire, points critiques, zones de surcharge ou compensation, etc.

🧠 Rédige ensuite un **feedback structuré et pédagogique** pour l'utilisateur final (non expert). Le ton doit être **bienveillant, motivant et clair**.

Structure recommandée du feedback :

1. 🧠 **Analyse rapide du mouvement global**
   - Décris ce que l'on voit (type de geste, niveau d'efficacité, stabilité générale).
   - Donne une première impression neutre.

2. ⚠️ **Points à corriger**
   - Liste les erreurs majeures détectées (ex : instabilité du bassin, angle de genou incorrect, bras trop rigide, perte d'équilibre).
   - Explique **les causes possibles** de ces erreurs (fatigue, manque de mobilité, défaut technique).
   - Décris les **risques associés** (blessures, performance réduite…).

3. ✅ **Ce que la personne fait bien**
   - Encourage ! Mentionne les bonnes postures, alignements, enchaînements réussis, rythme, respiration, etc.

4. 🏋️ **Conseils personnalisés**
   - Donne 2 à 4 recommandations pratiques, **simples et accessibles**, pour s'améliorer rapidement.
   - Ajoute si possible une **suggestion d'exercice ciblé** ou de correction immédiate (ex : renforcement du tronc, drill de mobilité de la cheville, travail devant un miroir...).

5. 📈 **Évolution à moyen terme**
   - Propose des axes de progression (amélioration de coordination, d'endurance posturale, de souplesse spécifique…).

---

🎯 Ta réponse doit tenir en **500 mots max**, être bien structurée avec des sections et des emoji pour l'accessibilité. Tu peux utiliser des termes techniques **avec explication simple** entre parenthèses.

Le texte doit être **adapté au grand public**, même s'il contient des notions expertes. Ton but est d'**aider la personne à progresser dès demain**, sans jargon inutile.

Merci !
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Tu es un coach expert en analyse du mouvement humain."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Erreur lors de la génération du feedback : {str(e)}")
        return f"""
## 🤖 Feedback automatique

**Type d'activité :** {activity_type}

**Observations techniques :** {observations}

**Note :** Le service de génération de feedback IA est temporairement indisponible. 
Voici un résumé basique de votre analyse :

- Les métriques de votre mouvement ont été calculées avec succès
- Consultez les observations techniques ci-dessus pour les détails
- Réessayez plus tard pour obtenir un feedback personnalisé complet

**Conseil général :** Continuez à pratiquer et concentrez-vous sur la régularité du mouvement !
"""