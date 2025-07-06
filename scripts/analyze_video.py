import cv2
import mediapipe as mp
import numpy as np
import math
import os

class UniversalMovementAnalyzer:
    def __init__(self):
        try:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception as e:
            print(f"Erreur d'initialisation MediaPipe : {e}")
            raise
        
    def analyze_video_internal(self, input_path, output_path, activity_type="general"):
        """Analyse universelle de mouvement avec codec web-compatible"""
        
        try:
            # Vérifier si le fichier d'entrée existe
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Fichier vidéo non trouvé : {input_path}")
            
            cap = cv2.VideoCapture(input_path)
            
            if not cap.isOpened():
                raise ValueError(f"Impossible d'ouvrir la vidéo : {input_path}")
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Vérifier les dimensions
            if width <= 0 or height <= 0:
                raise ValueError("Dimensions de vidéo invalides")
            
            # S'assurer que le fichier de sortie a l'extension .mp4
            if not output_path.endswith('.mp4'):
                output_path = os.path.splitext(output_path)[0] + '.mp4'
            
            # Essayer différents codecs dans l'ordre de préférence pour compatibilité web
            codecs_to_try = [
                ('H264', cv2.VideoWriter_fourcc(*'H264')),
                ('X264', cv2.VideoWriter_fourcc(*'X264')),
                ('avc1', cv2.VideoWriter_fourcc(*'avc1')),
                ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
                ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
                ('mp4v', cv2.VideoWriter_fourcc(*'mp4v'))  # fallback
            ]
            
            out = None
            used_codec = None
            
            for codec_name, fourcc in codecs_to_try:
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                if out.isOpened():
                    used_codec = codec_name
                    print(f"✅ Utilisation du codec {codec_name} pour compatibilité web")
                    break
                else:
                    out.release()
            
            if not out or not out.isOpened():
                raise ValueError("Aucun codec vidéo compatible trouvé")
            
            # Variables d'analyse
            frame_count = 0
            pose_data = []
            
            print(f"🎬 Début de l'analyse avec codec {used_codec}")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.pose.process(rgb_frame)
                    
                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        
                        # Analyse biomécanique
                        current_analysis = self.analyze_pose_frame(landmarks, width, height)
                        pose_data.append(current_analysis)
                        
                        # Dessiner les landmarks
                        self.mp_drawing.draw_landmarks(
                            frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255,255,255), thickness=2)
                        )
                        
                        # Afficher métriques en temps réel
                        self.display_metrics(frame, current_analysis, activity_type)
                    
                    out.write(frame)
                    
                except Exception as e:
                    print(f"Erreur lors du traitement de la frame {frame_count}: {e}")
                    # Continuer avec la frame suivante
                    out.write(frame)
                    continue
            
            cap.release()
            out.release()
            
            # Vérifier que le fichier de sortie est valide
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise ValueError("Le fichier vidéo de sortie est invalide ou vide")
            
            file_size = os.path.getsize(output_path)
            print(f"✅ Vidéo créée avec succès: {output_path}")
            print(f"📊 Codec: {used_codec}, Taille: {file_size/1024/1024:.1f}MB, Frames: {frame_count}")
            
            # Calculer métriques finales
            final_metrics = self.calculate_final_metrics(pose_data, activity_type)
            observations = self.generate_observations(final_metrics, activity_type)
            
            return {
                'observations': observations,
                'metrics': final_metrics,
                'activity_type': activity_type,
                'frame_count': frame_count,
                'codec_used': used_codec,
                'file_size': file_size
            }
            
        except Exception as e:
            print(f"❌ Erreur dans analyze_video_internal : {e}")
            # Nettoyer en cas d'erreur
            if 'out' in locals() and out:
                out.release()
            if 'cap' in locals() and cap:
                cap.release()
            
            # Retourner des résultats par défaut en cas d'erreur
            return {
                'observations': f"Erreur lors de l'analyse : {str(e)}",
                'metrics': {'overall_score': 0, 'error': True},
                'activity_type': activity_type,
                'frame_count': 0,
                'codec_used': 'error',
                'file_size': 0
            }
    
    def analyze_pose_frame(self, landmarks, width, height):
        """Analyse d'une frame individuelle avec gestion d'erreurs"""
        
        try:
            # Points clés avec vérification
            if len(landmarks) < 33:  # MediaPipe pose a 33 landmarks
                raise ValueError("Nombre insuffisant de landmarks détectés")
            
            nose = landmarks[0]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_elbow = landmarks[13]
            right_elbow = landmarks[14]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            left_knee = landmarks[25]
            right_knee = landmarks[26]
            left_ankle = landmarks[27]
            right_ankle = landmarks[28]
            
            analysis = {}
            
            # 1. ÉQUILIBRE (centre de gravité)
            center_of_mass = self.calculate_center_of_mass(landmarks)
            foot_center = ((left_ankle.x + right_ankle.x) / 2, (left_ankle.y + right_ankle.y) / 2)
            balance_offset = abs(center_of_mass[0] - foot_center[0])
            analysis['balance_score'] = max(0, min(100, 100 - (balance_offset * 1000)))
            
            # 2. SYMÉTRIE CORPORELLE
            shoulder_symmetry = abs(left_shoulder.y - right_shoulder.y)
            hip_symmetry = abs(left_hip.y - right_hip.y)
            analysis['symmetry_score'] = max(0, min(100, 100 - ((shoulder_symmetry + hip_symmetry) * 500)))
            
            # 3. ALIGNEMENT POSTURAL
            head_shoulder_alignment = abs(nose.x - ((left_shoulder.x + right_shoulder.x) / 2))
            shoulder_hip_alignment = abs(((left_shoulder.x + right_shoulder.x) / 2) - ((left_hip.x + right_hip.x) / 2))
            analysis['posture_score'] = max(0, min(100, 100 - ((head_shoulder_alignment + shoulder_hip_alignment) * 200)))
            
            # 4. ANGLES ARTICULAIRES
            analysis['left_elbow_angle'] = self.calculate_angle(left_shoulder, left_elbow, landmarks[15])
            analysis['right_elbow_angle'] = self.calculate_angle(right_shoulder, right_elbow, landmarks[16])
            analysis['left_knee_angle'] = self.calculate_angle(left_hip, left_knee, left_ankle)
            analysis['right_knee_angle'] = self.calculate_angle(right_hip, right_knee, right_ankle)
            
            # 5. STABILITÉ
            analysis['head_stability'] = 95  # Score de base
            
            # 6. AMPLITUDE DE MOUVEMENT
            analysis['shoulder_width'] = abs(left_shoulder.x - right_shoulder.x)
            analysis['hip_width'] = abs(left_hip.x - right_hip.x)
            
            return analysis
            
        except Exception as e:
            print(f"Erreur dans analyze_pose_frame : {e}")
            # Retourner des valeurs par défaut
            return {
                'balance_score': 50,
                'symmetry_score': 50,
                'posture_score': 50,
                'left_elbow_angle': 90,
                'right_elbow_angle': 90,
                'left_knee_angle': 90,
                'right_knee_angle': 90,
                'head_stability': 50,
                'shoulder_width': 0.3,
                'hip_width': 0.3,
                'error': True
            }
    
    def calculate_center_of_mass(self, landmarks):
        """Calcule le centre de masse approximatif"""
        weights = {'head': 0.08, 'torso': 0.46, 'arms': 0.12, 'legs': 0.34}
        
        head_pos = (landmarks[0].x, landmarks[0].y)
        torso_pos = ((landmarks[11].x + landmarks[12].x + landmarks[23].x + landmarks[24].x) / 4,
                     (landmarks[11].y + landmarks[12].y + landmarks[23].y + landmarks[24].y) / 4)
        
        com_x = head_pos[0] * weights['head'] + torso_pos[0] * weights['torso']
        com_y = head_pos[1] * weights['head'] + torso_pos[1] * weights['torso']
        
        return (com_x, com_y)
    
    def calculate_angle(self, a, b, c):
        """Calcule l'angle entre trois points avec gestion d'erreurs"""
        try:
            a_pos = np.array([a.x, a.y])
            b_pos = np.array([b.x, b.y])
            c_pos = np.array([c.x, c.y])
            
            ba = a_pos - b_pos
            bc = c_pos - b_pos
            
            # Vérifier les vecteurs nuls
            if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
                return 90.0
            
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            
            # Clamp pour éviter les erreurs d'arccos
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            angle = np.arccos(cosine_angle)
            
            return np.degrees(angle)
            
        except Exception as e:
            print(f"Erreur dans calculate_angle : {e}")
            return 90.0
    
    def display_metrics(self, frame, analysis, activity_type):
        """Affiche les métriques en temps réel sur la vidéo"""
        
        # Panneau de métriques avec fond semi-transparent amélioré
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (420, 220), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (10, 10), (420, 220), (255, 255, 255), 2)
        
        # Titre avec style amélioré
        cv2.putText(frame, f"ANALYSE: {activity_type.upper()}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Ligne de séparation
        cv2.line(frame, (20, 45), (400, 45), (255, 255, 255), 1)
        
        # Métriques avec couleurs adaptatives
        metrics = [
            ("Equilibre", analysis['balance_score']),
            ("Symetrie", analysis['symmetry_score']),
            ("Posture", analysis['posture_score']),
            ("Angle genou G", analysis['left_knee_angle'], "°"),
            ("Angle genou D", analysis['right_knee_angle'], "°"),
        ]
        
        for i, metric in enumerate(metrics):
            if len(metric) == 3:  # Avec unité
                name, value, unit = metric
                text = f"{name}: {value:.0f}{unit}"
            else:  # Sans unité
                name, value = metric
                text = f"{name}: {value:.0f}/100"
            
            # Couleur selon la performance
            if name.startswith("Angle"):
                color = (0, 255, 0)  # Vert pour les angles
            elif value > 80:
                color = (0, 255, 0)  # Vert excellent
            elif value > 60:
                color = (0, 255, 255)  # Jaune correct
            else:
                color = (0, 100, 255)  # Orange/Rouge à améliorer
            
            cv2.putText(frame, text, (20, 70 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Score global au bas
        overall = (analysis['balance_score'] + analysis['symmetry_score'] + analysis['posture_score']) / 3
        cv2.putText(frame, f"SCORE GLOBAL: {overall:.0f}/100", (20, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def calculate_final_metrics(self, pose_data, activity_type):
        """Calcule les métriques finales"""
        
        if not pose_data:
            return {'overall_score': 0, 'error': True}
            
        metrics = {}
        
        # Filtrer les frames avec erreur
        valid_frames = [frame for frame in pose_data if not frame.get('error', False)]
        
        if not valid_frames:
            return {'overall_score': 0, 'error': True}
        
        # Moyennes des scores
        metrics['avg_balance'] = np.mean([frame['balance_score'] for frame in valid_frames])
        metrics['avg_symmetry'] = np.mean([frame['symmetry_score'] for frame in valid_frames])
        metrics['avg_posture'] = np.mean([frame['posture_score'] for frame in valid_frames])
        
        # Variabilité (stabilité)
        balance_scores = [frame['balance_score'] for frame in valid_frames]
        posture_scores = [frame['posture_score'] for frame in valid_frames]
        
        metrics['balance_stability'] = max(0, 100 - np.std(balance_scores))
        metrics['posture_consistency'] = max(0, 100 - np.std(posture_scores))
        
        # Angles articulaires moyens
        metrics['avg_left_knee_angle'] = np.mean([frame['left_knee_angle'] for frame in valid_frames])
        metrics['avg_right_knee_angle'] = np.mean([frame['right_knee_angle'] for frame in valid_frames])
        metrics['avg_left_elbow_angle'] = np.mean([frame['left_elbow_angle'] for frame in valid_frames])
        metrics['avg_right_elbow_angle'] = np.mean([frame['right_elbow_angle'] for frame in valid_frames])
        
        # Score global
        metrics['overall_score'] = (metrics['avg_balance'] + metrics['avg_symmetry'] + metrics['avg_posture']) / 3
        
        # Informations supplémentaires
        metrics['valid_frames'] = len(valid_frames)
        metrics['total_frames'] = len(pose_data)
        metrics['detection_rate'] = (len(valid_frames) / len(pose_data)) * 100
        
        return metrics
    
    def generate_observations(self, metrics, activity_type):
        """Génère les observations textuelles pour l'IA"""
        
        if metrics.get('error', False):
            return "Erreur lors de l'analyse des métriques"
        
        observations = []
        
        # Évaluation globale
        overall = metrics.get('overall_score', 0)
        detection_rate = metrics.get('detection_rate', 0)
        
        if detection_rate < 50:
            observations.append(f"⚠️ Détection posturale partielle ({detection_rate:.0f}% des frames)")
        
        if overall > 85:
            observations.append("✅ Très bonne qualité de mouvement globale")
        elif overall > 70:
            observations.append("⚠️ Qualité de mouvement correcte avec améliorations possibles")
        else:
            observations.append("🔴 Qualité de mouvement nécessitant des corrections importantes")
        
        # Équilibre
        balance = metrics.get('avg_balance', 0)
        balance_stability = metrics.get('balance_stability', 0)
        
        if balance < 70:
            observations.append(f"⚖️ Déséquilibre détecté (score: {balance:.0f}/100)")
        else:
            observations.append(f"⚖️ Bon équilibre maintenu (score: {balance:.0f}/100)")
        
        if balance_stability < 70:
            observations.append(f"🌊 Instabilité importante (stabilité: {balance_stability:.0f}/100)")
        
        # Symétrie
        symmetry = metrics.get('avg_symmetry', 0)
        if symmetry < 75:
            observations.append(f"📐 Asymétrie corporelle notable (score: {symmetry:.0f}/100)")
        
        # Posture
        posture = metrics.get('avg_posture', 0)
        posture_consistency = metrics.get('posture_consistency', 0)
        
        if posture < 75:
            observations.append(f"🏗️ Alignement postural à améliorer (score: {posture:.0f}/100)")
        
        if posture_consistency < 70:
            observations.append(f"📊 Posture variable (constance: {posture_consistency:.0f}/100)")
        
        # Angles articulaires
        left_knee = metrics.get('avg_left_knee_angle', 0)
        right_knee = metrics.get('avg_right_knee_angle', 0)
        left_elbow = metrics.get('avg_left_elbow_angle', 0)
        right_elbow = metrics.get('avg_right_elbow_angle', 0)
        
        if abs(left_knee - right_knee) > 10:
            observations.append(f"🦵 Asymétrie des genoux (G:{left_knee:.0f}° vs D:{right_knee:.0f}°)")
        
        if abs(left_elbow - right_elbow) > 15:
            observations.append(f"💪 Asymétrie des coudes (G:{left_elbow:.0f}° vs D:{right_elbow:.0f}°)")
        
        # Spécificités par activité
        if activity_type == "squat":
            if left_knee < 90 or right_knee < 90:
                observations.append("📏 Amplitude de squat insuffisante")
            elif left_knee > 90 and right_knee > 90:
                observations.append("✅ Bonne amplitude de squat")
                
        elif activity_type == "yoga":
            if posture > 85:
                observations.append("🧘 Excellent maintien de la posture")
            if balance > 80:
                observations.append("⚖️ Stabilité remarquable en posture")
                
        elif activity_type == "équilibre":
            if balance > 80 and balance_stability > 75:
                observations.append("⚖️ Capacités d'équilibration excellentes")
            elif balance < 60:
                observations.append("🎯 Équilibre à travailler prioritairement")
                
        elif activity_type == "basketball":
            if left_elbow > 90 and right_elbow > 90:
                observations.append("🏀 Bonne extension des bras pour le tir")
                
        elif activity_type == "golf":
            if posture > 75 and symmetry > 75:
                observations.append("⛳ Bon alignement pour le swing")
        
        # Performance globale
        if overall > 80 and balance_stability > 75:
            observations.append("🌟 Performance globale de haute qualité")
        elif overall < 60:
            observations.append("💪 Potentiel d'amélioration significatif")
        
        return " | ".join(observations)

# Fonction principale pour compatibilité avec app.py
def analyze_video(input_path, output_path, activity_type="general"):
    """Fonction principale d'analyse - Interface pour app.py"""
    try:
        analyzer = UniversalMovementAnalyzer()
        result = analyzer.analyze_video_internal(input_path, output_path, activity_type)
        return result
    except Exception as e:
        print(f"❌ Erreur critique dans analyze_video : {e}")
        return {
            'observations': f"Erreur critique : {str(e)}",
            'metrics': {'overall_score': 0, 'error': True},
            'activity_type': activity_type,
            'frame_count': 0,
            'codec_used': 'error',
            'file_size': 0
        }