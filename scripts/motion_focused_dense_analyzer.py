import cv2
import numpy as np
import os

class MotionFocusedDenseAnalyzer:
    """Analyseur dense focalisé sur les zones de mouvement"""
    
    def __init__(self):
        self.setup_analyzers()
        self.previous_frame = None
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        
    def setup_analyzers(self):
        """Configuration des outils d'analyse"""
        try:
            self.sift = cv2.SIFT_create(nfeatures=300)
            self.optical_flow = cv2.optflow.createOptFlow_PCAFlow() if hasattr(cv2, 'optflow') else None
            print("✅ MotionFocusedDenseAnalyzer initialisé")
        except Exception as e:
            print(f"❌ Erreur initialisation: {e}")
            self.sift = None
    
    def detect_motion_zones(self, frame):
        """Détecte les zones de mouvement dans la frame"""
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_mask = np.zeros(gray.shape, dtype=np.uint8)
        
        try:
            # 1. DÉTECTION PAR SOUSTRACTION DE FOND
            fg_mask = self.background_subtractor.apply(frame)
            
            # 2. DÉTECTION PAR DIFFÉRENCE DE FRAMES
            if self.previous_frame is not None:
                frame_diff = cv2.absdiff(self.previous_frame, gray)
                _, diff_thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
                
                # Combiner les deux méthodes
                motion_mask = cv2.bitwise_or(fg_mask, diff_thresh)
            else:
                motion_mask = fg_mask
            
            # 3. NETTOYAGE DU MASQUE
            # Éliminer le bruit
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
            
            # Dilater pour agrandir les zones de mouvement
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            motion_mask = cv2.dilate(motion_mask, kernel_dilate, iterations=1)
            
            # 4. ZONES D'INTÉRÊT HUMAIN (si détection MediaPipe disponible)
            if hasattr(self, 'human_bbox') and self.human_bbox:
                x, y, w, h = self.human_bbox
                # Ajouter une zone rectangulaire autour du corps humain
                cv2.rectangle(motion_mask, (x-50, y-50), (x+w+50, y+h+50), 255, -1)
            
            self.previous_frame = gray.copy()
            
            return motion_mask
            
        except Exception as e:
            print(f"Erreur détection mouvement: {e}")
            # En cas d'erreur, retourner un masque complet
            return np.ones(gray.shape, dtype=np.uint8) * 255
    
    def set_human_detection_zone(self, landmarks, frame_width, frame_height):
        """Définit la zone de détection humaine basée sur MediaPipe"""
        
        if landmarks and len(landmarks) > 0:
            # Calculer la bounding box du corps humain
            x_coords = [lm.x * frame_width for lm in landmarks]
            y_coords = [lm.y * frame_height for lm in landmarks]
            
            min_x, max_x = int(min(x_coords)), int(max(x_coords))
            min_y, max_y = int(min(y_coords)), int(max(y_coords))
            
            # Ajouter une marge
            margin = 50
            self.human_bbox = (
                max(0, min_x - margin),
                max(0, min_y - margin),
                min(frame_width, max_x - min_x + 2*margin),
                min(frame_height, max_y - min_y + 2*margin)
            )
        else:
            self.human_bbox = None
    
    def extract_motion_focused_points(self, frame, motion_mask):
        """Extraction de points focalisés sur les zones de mouvement"""
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        points = []
        h, w = frame.shape[:2]
        
        # 1. POINTS SIFT DANS LES ZONES DE MOUVEMENT
        if self.sift is not None:
            try:
                sift_kp = self.sift.detect(gray, motion_mask)
                for i, kp in enumerate(sift_kp[:150]):  # Limité à 150 points
                    points.append({
                        'id': f'M{i:03d}',
                        'type': 'motion_sift',
                        'x': int(kp.pt[0]),
                        'y': int(kp.pt[1]),
                        'strength': kp.response,
                        'color': (0, 255, 255),  # Jaune pour mouvement
                        'size': 3
                    })
            except Exception as e:
                print(f"Erreur SIFT motion: {e}")
        
        # 2. COINS HARRIS DANS LES ZONES DE MOUVEMENT
        try:
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, 
                                            minDistance=15, mask=motion_mask)
            if corners is not None:
                for i, corner in enumerate(corners):
                    x, y = corner.ravel().astype(int)
                    points.append({
                        'id': f'C{i:03d}',
                        'type': 'motion_corner',
                        'x': x,
                        'y': y,
                        'color': (255, 100, 0),  # Orange pour coins
                        'size': 2
                    })
        except Exception as e:
            print(f"Erreur corners motion: {e}")
        
        # 3. POINTS DE FLUX OPTIQUE (si mouvement détecté)
        try:
            if self.previous_frame is not None:
                # Calculer le flux optique dense sur les zones de mouvement
                flow = cv2.calcOpticalFlowPyrLK(self.previous_frame, gray, None, None)
                
                # Échantillonner des points sur la grille dans les zones de mouvement
                step = 25
                flow_count = 0
                for y in range(step, h-step, step):
                    for x in range(step, w-step, step):
                        if motion_mask[y, x] > 0:  # Seulement dans les zones de mouvement
                            # Calculer la magnitude du mouvement local
                            roi = motion_mask[y-10:y+10, x-10:x+10]
                            if np.sum(roi) > 100:  # Seuil de mouvement
                                points.append({
                                    'id': f'F{flow_count:03d}',
                                    'type': 'optical_flow',
                                    'x': x,
                                    'y': y,
                                    'color': (255, 0, 255),  # Magenta pour flux
                                    'size': 2
                                })
                                flow_count += 1
        except Exception as e:
            print(f"Erreur flux optique: {e}")
        
        # 4. CONTOURS ACTIFS (objets en mouvement)
        try:
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_count = 0
            for contour in contours:
                if cv2.contourArea(contour) > 200:  # Filtrer les petits contours
                    # Centre du contour
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        points.append({
                            'id': f'O{contour_count:03d}',
                            'type': 'moving_object',
                            'x': cx,
                            'y': cy,
                            'area': cv2.contourArea(contour),
                            'color': (0, 255, 0),  # Vert pour objets
                            'size': 4
                        })
                        contour_count += 1
        except Exception as e:
            print(f"Erreur contours: {e}")
        
        return points
    
    def extract_motion_focused_lines(self, frame, motion_mask):
        """Extraction de lignes focalisées sur les zones de mouvement"""
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lines = []
        
        try:
            # Appliquer Canny seulement sur les zones de mouvement
            edges = cv2.Canny(gray, 50, 150)
            motion_edges = cv2.bitwise_and(edges, motion_mask)
            
            # Lignes Hough sur les zones de mouvement
            hough_lines = cv2.HoughLinesP(motion_edges, 1, np.pi/180, 
                                        threshold=30, minLineLength=20, maxLineGap=15)
            
            if hough_lines is not None:
                for i, line in enumerate(hough_lines[:30]):  # Limité à 30 lignes
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    
                    lines.append({
                        'id': f'ML{i:02d}',
                        'type': 'motion_line',
                        'start': (x1, y1),
                        'end': (x2, y2),
                        'length': length,
                        'color': (255, 255, 0),  # Jaune pour lignes de mouvement
                        'thickness': 2
                    })
        except Exception as e:
            print(f"Erreur lignes motion: {e}")
        
        # Lignes de direction de mouvement
        try:
            if hasattr(self, 'human_bbox') and self.human_bbox:
                x, y, w, h = self.human_bbox
                center_x, center_y = x + w//2, y + h//2
                
                # Lignes de référence autour du corps humain
                reference_lines = [
                    {'id': 'REF1', 'type': 'reference', 'start': (center_x-50, center_y), 
                     'end': (center_x+50, center_y), 'color': (0, 255, 255), 'thickness': 1},
                    {'id': 'REF2', 'type': 'reference', 'start': (center_x, center_y-50), 
                     'end': (center_x, center_y+50), 'color': (0, 255, 255), 'thickness': 1},
                ]
                lines.extend(reference_lines)
        except Exception as e:
            print(f"Erreur lignes référence: {e}")
        
        return lines
    
    def draw_motion_focused_annotations(self, frame, points, lines, motion_mask=None):
        """Dessine les annotations focalisées sur le mouvement"""
        
        annotated_frame = frame.copy()
        
        # 1. OPTIONNEL : Afficher les zones de mouvement (semi-transparent)
        if motion_mask is not None and hasattr(self, 'show_motion_zones') and self.show_motion_zones:
            motion_overlay = cv2.applyColorMap(motion_mask, cv2.COLORMAP_JET)
            cv2.addWeighted(annotated_frame, 0.8, motion_overlay, 0.2, 0, annotated_frame)
        
        # 2. DESSINER LES LIGNES DE MOUVEMENT
        for line in lines:
            thickness = line.get('thickness', 1)
            if line['type'] == 'motion_line':
                cv2.line(annotated_frame, line['start'], line['end'], line['color'], thickness)
            elif line['type'] == 'reference':
                self.draw_dashed_line(annotated_frame, line['start'], line['end'], line['color'])
            
            # ID de ligne (plus discret)
            if line['type'] != 'reference':
                mid_x = (line['start'][0] + line['end'][0]) // 2
                mid_y = (line['start'][1] + line['end'][1]) // 2
                cv2.putText(annotated_frame, line['id'], (mid_x, mid_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, line['color'], 1)
        
        # 3. DESSINER LES POINTS DE MOUVEMENT
        for point in points:
            size = point.get('size', 2)
            
            # Point principal selon le type
            if point['type'] == 'motion_sift':
                cv2.circle(annotated_frame, (point['x'], point['y']), size, point['color'], -1)
                cv2.circle(annotated_frame, (point['x'], point['y']), size+2, point['color'], 1)
            elif point['type'] == 'moving_object':
                cv2.circle(annotated_frame, (point['x'], point['y']), size, point['color'], -1)
                # Croix pour marquer le centre
                cv2.line(annotated_frame, (point['x']-5, point['y']), (point['x']+5, point['y']), point['color'], 2)
                cv2.line(annotated_frame, (point['x'], point['y']-5), (point['x'], point['y']+5), point['color'], 2)
            else:
                cv2.circle(annotated_frame, (point['x'], point['y']), size, point['color'], -1)
            
            # ID du point (style professionnel)
            cv2.putText(annotated_frame, point['id'], 
                       (point['x'] + 8, point['y'] - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, point['color'], 1)
        
        # 4. STATISTIQUES FOCALISÉES
        self.draw_motion_stats(annotated_frame, len(points), len(lines), motion_mask)
        
        return annotated_frame
    
    def draw_dashed_line(self, frame, start, end, color):
        """Dessine une ligne pointillée"""
        x1, y1 = start
        x2, y2 = end
        length = int(np.sqrt((x2-x1)**2 + (y2-y1)**2))
        
        if length == 0:
            return
        
        dash_length = 5
        gap_length = 3
        total_dash = dash_length + gap_length
        
        dx = (x2 - x1) / length
        dy = (y2 - y1) / length
        
        for i in range(0, length, total_dash):
            start_x = int(x1 + i * dx)
            start_y = int(y1 + i * dy)
            end_dash = min(i + dash_length, length)
            end_x = int(x1 + end_dash * dx)
            end_y = int(y1 + end_dash * dy)
            cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 1)
    
    def draw_motion_stats(self, frame, num_points, num_lines, motion_mask):
        """Panneau de statistiques focalisées sur le mouvement"""
        
        h, w = frame.shape[:2]
        panel_x = w - 250
        panel_y = 10
        panel_w = 230
        panel_h = 140
        
        # Fond du panneau
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Bordure
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 255, 255), 2)
        
        # Titre
        cv2.putText(frame, "ANALYSE MOTION-FOCUSED", (panel_x + 10, panel_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Ligne de séparation
        cv2.line(frame, (panel_x + 10, panel_y + 25), (panel_x + panel_w - 10, panel_y + 25), (255, 255, 255), 1)
        
        # Calcul du pourcentage de mouvement
        motion_percentage = 0
        if motion_mask is not None:
            motion_pixels = np.sum(motion_mask > 0)
            total_pixels = motion_mask.shape[0] * motion_mask.shape[1]
            motion_percentage = (motion_pixels / total_pixels) * 100
        
        # Stats détaillées
        stats = [
            f"Points detectes: {num_points}",
            f"Lignes mouvement: {num_lines}",
            f"Zone active: {motion_percentage:.1f}%",
            f"Mode: MOTION FOCUS",
            f"Status: {'ACTIF' if num_points > 10 else 'STATIQUE'}"
        ]
        
        colors = [(255, 255, 255), (255, 255, 255), (0, 255, 0) if motion_percentage > 5 else (255, 100, 100),
                 (255, 255, 0), (0, 255, 0) if num_points > 10 else (255, 100, 100)]
        
        for i, (stat, color) in enumerate(zip(stats, colors)):
            cv2.putText(frame, stat, (panel_x + 10, panel_y + 45 + i * 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def analyze_motion_dense(self, frame, landmarks=None):
        """Analyse dense focalisée sur le mouvement - méthode principale"""
        
        # 1. Définir la zone humaine si landmarks disponibles
        if landmarks:
            h, w = frame.shape[:2]
            self.set_human_detection_zone(landmarks, w, h)
        
        # 2. Détecter les zones de mouvement
        motion_mask = self.detect_motion_zones(frame)
        
        # 3. Extraire points et lignes focalisés sur le mouvement
        motion_points = self.extract_motion_focused_points(frame, motion_mask)
        motion_lines = self.extract_motion_focused_lines(frame, motion_mask)
        
        # 4. Dessiner les annotations
        annotated_frame = self.draw_motion_focused_annotations(frame, motion_points, motion_lines, motion_mask)
        
        return annotated_frame, len(motion_points), len(motion_lines)