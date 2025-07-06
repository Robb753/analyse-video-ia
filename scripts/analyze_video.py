import cv2
import mediapipe as mp
import numpy as np
import math

class UniversalMovementAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def analyze_video_internal(self, input_path, output_path, activity_type="general"):
        """Analyse universelle de mouvement"""
        
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Variables d'analyse
        frame_count = 0
        pose_data = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
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
        
        cap.release()
        out.release()
        
        # Calculer métriques finales
        final_metrics = self.calculate_final_metrics(pose_data, activity_type)
        observations = self.generate_observations(final_metrics, activity_type)
        
        return {
            'observations': observations,
            'metrics': final_metrics,
            'activity_type': activity_type,
            'frame_count': frame_count
        }
    
    def analyze_pose_frame(self, landmarks, width, height):
        """Analyse d'une frame individuelle"""
        
        # Points clés
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
        analysis['balance_score'] = max(0, 100 - (balance_offset * 1000))
        
        # 2. SYMÉTRIE CORPORELLE
        shoulder_symmetry = abs(left_shoulder.y - right_shoulder.y)
        hip_symmetry = abs(left_hip.y - right_hip.y)
        analysis['symmetry_score'] = max(0, 100 - ((shoulder_symmetry + hip_symmetry) * 500))
        
        # 3. ALIGNEMENT POSTURAL
        head_shoulder_alignment = abs(nose.x - ((left_shoulder.x + right_shoulder.x) / 2))
        shoulder_hip_alignment = abs(((left_shoulder.x + right_shoulder.x) / 2) - ((left_hip.x + right_hip.x) / 2))
        analysis['posture_score'] = max(0, 100 - ((head_shoulder_alignment + shoulder_hip_alignment) * 200))
        
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
        """Calcule l'angle entre trois points"""
        a_pos = np.array([a.x, a.y])
        b_pos = np.array([b.x, b.y])
        c_pos = np.array([c.x, c.y])
        
        ba = a_pos - b_pos
        bc = c_pos - b_pos
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def display_metrics(self, frame, analysis, activity_type):
        """Affiche les métriques en temps réel sur la vidéo"""
        
        # Panneau de métriques
        cv2.rectangle(frame, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 200), (255, 255, 255), 2)
        
        # Titre
        cv2.putText(frame, f"Analyse: {activity_type.upper()}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Métriques
        metrics = [
            f"Equilibre: {analysis['balance_score']:.0f}/100",
            f"Symetrie: {analysis['symmetry_score']:.0f}/100", 
            f"Posture: {analysis['posture_score']:.0f}/100",
            f"Angle genou G: {analysis['left_knee_angle']:.0f}°",
            f"Angle genou D: {analysis['right_knee_angle']:.0f}°",
        ]
        
        for i, metric in enumerate(metrics):
            color = (0, 255, 0) if analysis['balance_score'] > 80 else (0, 255, 255)
            cv2.putText(frame, metric, (20, 60 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def calculate_final_metrics(self, pose_data, activity_type):
        """Calcule les métriques finales"""
        
        if not pose_data:
            return {}
            
        metrics = {}
        
        # Moyennes des scores
        metrics['avg_balance'] = np.mean([frame['balance_score'] for frame in pose_data])
        metrics['avg_symmetry'] = np.mean([frame['symmetry_score'] for frame in pose_data])
        metrics['avg_posture'] = np.mean([frame['posture_score'] for frame in pose_data])
        
        # Variabilité
        metrics['balance_stability'] = 100 - np.std([frame['balance_score'] for frame in pose_data])
        metrics['posture_consistency'] = 100 - np.std([frame['posture_score'] for frame in pose_data])
        
        # Angles articulaires moyens
        metrics['avg_left_knee_angle'] = np.mean([frame['left_knee_angle'] for frame in pose_data])
        metrics['avg_right_knee_angle'] = np.mean([frame['right_knee_angle'] for frame in pose_data])
        
        # Score global
        metrics['overall_score'] = (metrics['avg_balance'] + metrics['avg_symmetry'] + metrics['avg_posture']) / 3
        
        return metrics
    
    def generate_observations(self, metrics, activity_type):
        """Génère les observations textuelles pour OpenAI"""
        
        observations = []
        
        # Évaluation globale
        overall = metrics.get('overall_score', 0)
        if overall > 85:
            observations.append("✅ Très bonne qualité de mouvement globale")
        elif overall > 70:
            observations.append("⚠️ Qualité de mouvement correcte avec améliorations possibles")
        else:
            observations.append("🔴 Qualité de mouvement nécessitant des corrections importantes")
        
        # Équilibre
        balance = metrics.get('avg_balance', 0)
        if balance < 70:
            observations.append(f"⚖️ Déséquilibre détecté (score: {balance:.0f}/100)")
        else:
            observations.append(f"⚖️ Bon équilibre maintenu (score: {balance:.0f}/100)")
        
        # Symétrie
        symmetry = metrics.get('avg_symmetry', 0)
        if symmetry < 75:
            observations.append(f"📐 Asymétrie corporelle notable (score: {symmetry:.0f}/100)")
        
        # Posture
        posture = metrics.get('avg_posture', 0)
        if posture < 75:
            observations.append(f"🏗️ Alignement postural à améliorer (score: {posture:.0f}/100)")
        
        # Angles articulaires
        left_knee = metrics.get('avg_left_knee_angle', 0)
        right_knee = metrics.get('avg_right_knee_angle', 0)
        if abs(left_knee - right_knee) > 10:
            observations.append(f"🦵 Asymétrie des genoux (G:{left_knee:.0f}° vs D:{right_knee:.0f}°)")
        
        # Stabilité
        stability = metrics.get('balance_stability', 0)
        if stability < 70:
            observations.append("🌪️ Instabilité/oscillations importantes")
        
        # Spécificités par activité
        if activity_type == "squat":
            if left_knee < 90 or right_knee < 90:
                observations.append("📏 Amplitude de squat insuffisante")
        elif activity_type == "yoga":
            if posture > 85:
                observations.append("🧘 Excellent maintien de la posture")
        elif activity_type == "équilibre":
            if balance > 80 and stability > 75:
                observations.append("⚖️ Capacités d'équilibration excellentes")
        
        return " | ".join(observations)

# Fonction principale pour compatibilité avec ton app.py
def analyze_video(input_path, output_path, activity_type="general"):
    """Fonction principale d'analyse - Interface pour app.py"""
    analyzer = UniversalMovementAnalyzer()
    result = analyzer.analyze_video_internal(input_path, output_path, activity_type)
    return result