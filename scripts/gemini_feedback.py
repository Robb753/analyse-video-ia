import google.generativeai as genai
import os
from typing import Dict, Optional

# Configuration Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def generate_feedback(activity_type: str, observations: str, metrics: Optional[Dict] = None) -> str:
    """
    Génère un feedback personnalisé avec Gemini
    
    Args:
        activity_type (str): Type d'activité analysée
        observations (str): Observations techniques de l'analyse
        metrics (dict): Métriques détaillées (optionnel)
    
    Returns:
        str: Feedback formaté pour l'utilisateur
    """
    
    # Préparer les métriques si disponibles
    metrics_text = ""
    if metrics and not metrics.get('error', False):
        metrics_text = f"""
        
📊 **Métriques mesurées :**
- Score global : {metrics.get('overall_score', 0):.1f}/100
- Équilibre : {metrics.get('avg_balance', 0):.1f}/100
- Symétrie : {metrics.get('avg_symmetry', 0):.1f}/100
- Posture : {metrics.get('avg_posture', 0):.1f}/100
- Stabilité : {metrics.get('balance_stability', 0):.1f}/100
        """
    
    # Adapter le prompt selon l'activité
    activity_context = {
        "basketball": "tir au basketball, focus sur l'alignement du bras, l'équilibre des jambes et le suivi du geste",
        "golf": "swing de golf, analyse de la rotation du tronc, transfert de poids et tempo",
        "yoga": "posture de yoga, évaluation de l'alignement, stabilité et respiration",
        "squat": "exercice de squat, vérification de la profondeur, alignement genoux/hanches",
        "équilibre": "test d'équilibre, mesure de la stabilité posturale et contrôle moteur",
        "autre": "mouvement général, analyse biomécanique globale"
    }
    
    context = activity_context.get(activity_type, activity_context["autre"])
    
    prompt = f"""
Tu es un coach sportif expert en biomécanique. Analyse cette activité de **{activity_type}** ({context}).

**Observations automatiques détectées :**
{observations}

{metrics_text}

**Consigne :** Rédige un feedback **constructif et motivant** de 300 mots maximum, structuré ainsi :

## 🎯 Vue d'ensemble
Résume rapidement ce qui a été observé dans le mouvement.

## ✅ Points positifs  
Liste ce qui est bien fait (même si imparfait, trouve du positif !).

## ⚠️ Axes d'amélioration
Identifie 2-3 points techniques précis à corriger, avec l'impact sur la performance.

## 🏋️ Conseils pratiques
Donne 2-3 exercices ou corrections simples à appliquer immédiatement.

**Ton :** Bienveillant, technique mais accessible, motivant. Utilise des emojis pour structurer.
"""

    try:
        # Configuration du modèle
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Génération avec paramètres optimisés
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=400,
                top_p=0.8,
            )
        )
        
        return response.text
        
    except Exception as e:
        print(f"❌ Erreur Gemini : {str(e)}")
        
        # Fallback intelligent basé sur les métriques
        return generate_fallback_feedback(activity_type, observations, metrics)

def generate_fallback_feedback(activity_type: str, observations: str, metrics: Optional[Dict]) -> str:
    """Génère un feedback de secours basé sur les métriques"""
    
    if not metrics or metrics.get('error', False):
        return f"""
## 🎯 Analyse de votre {activity_type}

**Statut :** Analyse technique en cours

**Observations :** {observations}

## 💡 Conseils généraux
- Concentrez-vous sur la fluidité du mouvement
- Maintenez une posture droite et équilibrée  
- Respirez de manière contrôlée pendant l'exercice
- Pratiquez régulièrement pour améliorer la coordination

## 🔄 Prochaines étapes
Réessayez l'analyse avec une vidéo en meilleure qualité pour obtenir un feedback plus détaillé !
"""
    
    # Analyse des scores
    overall = metrics.get('overall_score', 0)
    balance = metrics.get('avg_balance', 0)
    symmetry = metrics.get('avg_symmetry', 0)
    posture = metrics.get('avg_posture', 0)
    
    # Feedback basé sur les scores
    feedback = f"""
## 🎯 Analyse de votre {activity_type}

**Score global :** {overall:.1f}/100

## ✅ Points positifs
"""
    
    if balance > 70:
        feedback += "- Bon équilibre général maintenu\n"
    if symmetry > 75:
        feedback += "- Symétrie corporelle correcte\n"
    if posture > 70:
        feedback += "- Alignement postural satisfaisant\n"
    
    feedback += "\n## ⚠️ Axes d'amélioration\n"
    
    if balance < 70:
        feedback += "- Travaillez votre équilibre avec des exercices sur une jambe\n"
    if symmetry < 75:
        feedback += "- Améliorez la symétrie en pratiquant devant un miroir\n"
    if posture < 70:
        feedback += "- Renforcez votre tronc pour une meilleure posture\n"
    
    feedback += f"""
## 🏋️ Conseils pour {activity_type}
- Échauffez-vous bien avant chaque session
- Concentrez-vous sur la qualité plutôt que la quantité
- Filmez-vous régulièrement pour suivre vos progrès
- Hydratez-vous et récupérez bien entre les entraînements
"""
    
    return feedback