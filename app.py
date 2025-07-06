from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
import os
import threading
import time
import schedule
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from scripts.analyze_video import analyze_video
from scripts.gemini_feedback import generate_feedback

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'mp4', 'mov'}

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Dictionnaire pour stocker le progrès des analyses
analysis_progress = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_files(task_id):
    """Nettoie les fichiers associés à une tâche"""
    try:
        if task_id in analysis_progress:
            task_data = analysis_progress[task_id]
            
            # Récupère les chemins des fichiers
            filename = task_data.get('filename', '')
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            output_filename = task_data.get('output_filename', '')
            output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
            
            # Supprime le fichier d'entrée (upload)
            if os.path.exists(input_path):
                os.remove(input_path)
                print(f"🗑️ Supprimé: {input_path}")
            
            # Supprime le fichier de sortie (processed) - optionnel
            if os.path.exists(output_path):
                os.remove(output_path)
                print(f"🗑️ Supprimé: {output_path}")
            
            # Supprime les données de progression
            del analysis_progress[task_id]
            print(f"🗑️ Tâche nettoyée: {task_id}")
            
    except Exception as e:
        print(f"❌ Erreur nettoyage {task_id}: {str(e)}")

def cleanup_old_files():
    """Nettoie automatiquement les anciens fichiers (>24h)"""
    try:
        current_time = time.time()
        cutoff_time = current_time - (24 * 60 * 60)  # 24 heures
        
        # Nettoie les dossiers uploads et processed
        for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    if os.path.isfile(file_path):
                        file_time = os.path.getctime(file_path)
                        if file_time < cutoff_time:
                            os.remove(file_path)
                            print(f"🗑️ Fichier ancien supprimé: {file_path}")
        
        # Nettoie les tâches anciennes en mémoire
        to_remove = []
        for task_id, task_data in analysis_progress.items():
            upload_time = task_data.get('upload_time', current_time)
            if upload_time < cutoff_time:
                to_remove.append(task_id)
        
        for task_id in to_remove:
            del analysis_progress[task_id]
            print(f"🗑️ Tâche ancienne supprimée: {task_id}")
            
    except Exception as e:
        print(f"❌ Erreur nettoyage automatique: {str(e)}")

def analyze_video_with_progress(input_path, output_path, task_id):
    """Fonction qui lance l'analyse avec suivi du progrès"""
    try:
        analysis_progress[task_id].update({
            'status': 'processing',
            'progress': 0,
            'message': 'Initialisation...'
        })

        # Étapes de progression
        steps = [
            {'progress': 10, 'message': 'Lecture de la vidéo...'},
            {'progress': 25, 'message': 'Initialisation MediaPipe...'},
            {'progress': 40, 'message': 'Détection des poses...'},
            {'progress': 60, 'message': 'Analyse biomécanique...'},
            {'progress': 75, 'message': 'Calcul des métriques...'},
            {'progress': 85, 'message': 'Génération des annotations...'},
            {'progress': 95, 'message': 'Finalisation...'},
        ]

        for step in steps:
            analysis_progress[task_id].update(step)
            time.sleep(0.5)

        # Récupère le type d'activité
        activity_type = analysis_progress[task_id].get('activity_type', 'autre')
        
        analysis_progress[task_id].update({
            'progress': 45,
            'message': 'Analyse biomécanique en cours...'
        })
        
        analysis_results = analyze_video(input_path, output_path, activity_type)
        
        analysis_progress[task_id].update({
            'progress': 80,
            'message': 'Génération du feedback IA...'
        })
        
        # Nouvelles observations basées sur l'analyse RÉELLE
        observations = analysis_results.get('observations', 'Analyse biomécanique effectuée')
        
        # Ajoute des métriques détaillées
        metrics = analysis_results.get('metrics', {})
        if metrics:
            additional_info = []
            if 'overall_score' in metrics:
                additional_info.append(f"Score global: {metrics['overall_score']:.1f}/100")
            if 'avg_balance' in metrics:
                additional_info.append(f"Équilibre moyen: {metrics['avg_balance']:.1f}/100")
            if 'avg_symmetry' in metrics:
                additional_info.append(f"Symétrie: {metrics['avg_symmetry']:.1f}/100")
            
            if additional_info:
                observations += " | " + " | ".join(additional_info)
        
        # Génère le feedback
        feedback = generate_feedback(activity_type, observations, metrics)
        
        # Stocke les résultats complets + programme le nettoyage
        analysis_progress[task_id].update({
            'feedback': feedback,
            'observations': observations,
            'metrics': metrics,
            'analysis_summary': {
                'activity_type': activity_type,
                'frames_analyzed': analysis_results.get('frame_count', 0),
                'overall_score': metrics.get('overall_score', 0) if metrics else 0,
                'key_findings': observations.split(' | ')[:3]
            },
            'cleanup_scheduled': False,  # Flag pour éviter double nettoyage
            'result_viewed': False  # Flag pour savoir si l'utilisateur a vu le résultat
        })

        # Finalise la tâche
        analysis_progress[task_id].update({
            'status': 'completed',
            'progress': 100,
            'message': 'Analyse terminée !',
            'output_filename': os.path.basename(output_path),
            'completion_time': time.time()
        })

        # Programme le nettoyage automatique après 2 heures
        threading.Timer(2 * 60 * 60, cleanup_files, args=[task_id]).start()

    except Exception as e:
        print(f"Erreur dans analyze_video_with_progress: {str(e)}")
        analysis_progress[task_id] = {
            'status': 'error',
            'progress': 0,
            'message': f'Erreur: {str(e)}'
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    print(f"Requête reçue: {request.method}")
    print(f"Fichiers dans la requête: {list(request.files.keys())}")
    print(f"Données du formulaire: {dict(request.form)}")
    
    if 'video' not in request.files:
        print("❌ Aucun fichier vidéo dans la requête")
        flash('Aucun fichier vidéo envoyé.')
        return redirect(url_for('index'))

    file = request.files['video']
    if file.filename == '':
        print("❌ Nom de fichier vide")
        flash('Fichier non sélectionné.')
        return redirect(url_for('index'))

    if not file or not allowed_file(file.filename):
        print(f"❌ Fichier non autorisé: {file.filename if file else 'None'}")
        flash('Format de fichier non autorisé.')
        return redirect(url_for('index'))

    try:
        activity_type = request.form.get('activity_type', 'autre')
        print(f"✅ Type d'activité: {activity_type}")
        
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], f"processed_{filename}")
        
        print(f"💾 Sauvegarde du fichier: {input_path}")
        file.save(input_path)
        
        # Vérifier que le fichier a bien été sauvegardé
        if not os.path.exists(input_path):
            print("❌ Erreur: fichier non sauvegardé")
            flash('Erreur lors de la sauvegarde du fichier.')
            return redirect(url_for('index'))
        
        print(f"✅ Fichier sauvegardé avec succès. Taille: {os.path.getsize(input_path)} bytes")

        # ID de tâche unique
        task_id = f"task_{int(time.time())}_{filename}"
        print(f"🆔 Task ID: {task_id}")

        # Initialise l'entrée de suivi
        analysis_progress[task_id] = {
            'status': 'initializing',
            'progress': 0,
            'message': 'Préparation...',
            'activity_type': activity_type,
            'filename': filename,
            'upload_time': time.time(),
            'input_path': input_path,
            'output_path': output_path
        }

        # Lancer le traitement dans un thread
        thread = threading.Thread(
            target=analyze_video_with_progress,
            args=(input_path, output_path, task_id)
        )
        thread.daemon = True
        thread.start()
        
        print(f"🚀 Thread d'analyse démarré pour {task_id}")
        return redirect(url_for('progress', task_id=task_id))
        
    except Exception as e:
        print(f"❌ Erreur lors du traitement: {str(e)}")
        flash(f'Erreur lors du traitement: {str(e)}')
        return redirect(url_for('index'))

@app.route('/progress/<task_id>')
def progress(task_id):
    return render_template('progress.html', task_id=task_id)

@app.route('/api/progress/<task_id>')
def get_progress(task_id):
    if task_id in analysis_progress:
        progress_data = analysis_progress[task_id].copy()
        
        if progress_data.get('status') == 'completed':
            progress_data['has_detailed_results'] = True
        
        return jsonify(progress_data)
    else:
        return jsonify({'status': 'not_found', 'progress': 0, 'message': 'Tâche non trouvée'})

@app.route('/result/<task_id>')
def result(task_id):
    if task_id in analysis_progress and analysis_progress[task_id]['status'] == 'completed':
        task_data = analysis_progress[task_id]
        filename = task_data['output_filename']
        activity_type = task_data.get('activity_type', 'non spécifié')
        feedback = task_data.get('feedback', 'Aucun feedback généré.')
        observations = task_data.get('observations', 'Aucune observation.')
        metrics = task_data.get('metrics', {})
        analysis_summary = task_data.get('analysis_summary', {})
        
        # Marque que l'utilisateur a vu le résultat
        analysis_progress[task_id]['result_viewed'] = True
        analysis_progress[task_id]['result_view_time'] = time.time()
        
        return render_template(
            'result.html',
            filename=filename,
            task_id=task_id,
            activity_type=activity_type,
            feedback=feedback,
            observations=observations,
            metrics=metrics,
            analysis_summary=analysis_summary
        )
    else:
        flash('Analyse non trouvée ou non terminée.')
        return redirect(url_for('index'))

@app.route('/processed/<filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/api/analysis/<task_id>')
def get_detailed_analysis(task_id):
    """API pour récupérer les détails complets de l'analyse"""
    if task_id in analysis_progress and analysis_progress[task_id]['status'] == 'completed':
        task_data = analysis_progress[task_id]
        return jsonify({
            'success': True,
            'activity_type': task_data.get('activity_type'),
            'observations': task_data.get('observations'),
            'metrics': task_data.get('metrics'),
            'feedback': task_data.get('feedback'),
            'analysis_summary': task_data.get('analysis_summary')
        })
    else:
        return jsonify({'success': False, 'message': 'Analyse non trouvée'})

@app.route('/api/cleanup/<task_id>', methods=['POST'])
def manual_cleanup(task_id):
    """Permet à l'utilisateur de nettoyer manuellement ses fichiers"""
    try:
        cleanup_files(task_id)
        return jsonify({'success': True, 'message': 'Fichiers supprimés avec succès'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Erreur: {str(e)}'})

# Démarre le nettoyage automatique
def start_cleanup_scheduler():
    """Démarre le planificateur de nettoyage automatique"""
    schedule.every().hour.do(cleanup_old_files)
    
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(60)  # Vérifie toutes les minutes
    
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()

if __name__ == '__main__':
    # Nettoyage initial au démarrage
    cleanup_old_files()
    
    # Démarre le planificateur
    start_cleanup_scheduler()
    
    app.run(debug=True)