# scripts/dense_analyzer.py
import cv2
import numpy as np
import google.generativeai as genai
from PIL import Image
import os

class DenseVisualAnalyzer:
    """Analyseur visuel dense - Version simplifiée pour intégration"""
    
    def __init__(self):
        self.setup_analyzers()
        
    def setup_analyzers(self):
        """Configuration des outils d'analyse"""
        # Détecteurs de features
        self.sift = cv2.SIFT_create(nfeatures=200)  # Réduit pour performance
        
    def extract_dense_points(self, frame):
        """Extraction de points d'intérêt multiples"""
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        points = []
        
        # 1. SIFT Features (points d'intérêt robustes)
        try:
            sift_kp = self.sift.detect(gray, None)
            for i, kp in enumerate(sift_kp[:100]):  # Top 100 points
                points.append({
                    'id': f'S{i}',
                    'type': 'sift',
                    'x': int(kp.pt[0]),
                    'y': int(kp.pt[1]),
                    'strength': kp.response,
                    'color': (0, 255, 255)  # Jaune
                })
        except Exception as e:
            print(f"Erreur SIFT: {e}")
        
        # 2. Coins Harris
        try:
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=50, qualityLevel=0.01, minDistance=10)
            if corners is not None:
                for i, corner in enumerate(corners):
                    x, y = corner.ravel().astype(int)
                    points.append({
                        'id': f'C{i}',
                        'type': 'corner',
                        'x': x,
                        'y': y,
                        'color': (255, 0, 0)  # Rouge
                    })
        except Exception as e:
            print(f"Erreur corners: {e}")
        
        # 3. Points sur grille d'analyse
        h, w = frame.shape[:2]
        grid_spacing = 60
        for y in range(0, h, grid_spacing):
            for x in range(0, w, grid_spacing):
                # Calculer intérêt local
                roi = gray[max(0, y-15):min(h, y+15), max(0, x-15):min(w, x+15)]
                if roi.size > 0:
                    interest = np.var(roi)
                    if interest > 50:  # Seuil d'intérêt
                        points.append({
                            'id': f'G{len(points)}',
                            'type': 'grid',
                            'x': x,
                            'y': y,
                            'interest': float(interest),
                            'color': (100, 255, 100)  # Vert clair
                        })
        
        return points
    
    def extract_dense_lines(self, frame):
        """Extraction de lignes géométriques"""
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lines = []
        
        try:
            # Détection de contours
            edges = cv2.Canny(gray, 50, 150)
            hough_lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
            
            if hough_lines is not None:
                for i, line in enumerate(hough_lines[:50]):  # Limiter à 50 lignes
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    
                    lines.append({
                        'id': f'L{i}',
                        'type': 'hough',
                        'start': (x1, y1),
                        'end': (x2, y2),
                        'length': length,
                        'color': (255, 255, 255)  # Blanc
                    })
        except Exception as e:
            print(f"Erreur lignes: {e}")
        
        return lines
    
    def draw_dense_annotations(self, frame, points, lines):
        """Dessine toutes les annotations denses sur la frame"""
        
        annotated_frame = frame.copy()
        
        # 1. DESSINER TOUS LES POINTS
        for point in points:
            # Point principal
            cv2.circle(annotated_frame, (point['x'], point['y']), 3, point['color'], -1)
            
            # ID du point (style comme vos images de référence)
            cv2.putText(annotated_frame, point['id'], 
                       (point['x'] + 5, point['y'] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, point['color'], 1)
        
        # 2. DESSINER TOUTES LES LIGNES
        for line in lines:
            cv2.line(annotated_frame, line['start'], line['end'], line['color'], 1)
            
            # ID de la ligne au milieu
            mid_x = (line['start'][0] + line['end'][0]) // 2
            mid_y = (line['start'][1] + line['end'][1]) // 2
            cv2.putText(annotated_frame, line['id'], 
                       (mid_x, mid_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, line['color'], 1)
        
        # 3. STATISTIQUES DENSE
        self.draw_dense_stats(annotated_frame, len(points), len(lines))
        
        return annotated_frame
    
    def draw_dense_stats(self, frame, num_points, num_lines):
        """Panneau de statistiques dense"""
        
        # Panneau en haut à droite
        h, w = frame.shape[:2]
        panel_x = w - 200
        panel_y = 10
        
        # Fond du panneau
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + 180, panel_y + 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + 180, panel_y + 100), (255, 255, 255), 1)
        
        # Titre
        cv2.putText(frame, "ANALYSE DENSE", (panel_x + 5, panel_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Stats
        cv2.putText(frame, f"Points: {num_points}", (panel_x + 5, panel_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Lignes: {num_lines}", (panel_x + 5, panel_y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Total: {num_points + num_lines}", (panel_x + 5, panel_y + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)