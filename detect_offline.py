import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
import pygame
import time
import threading
from datetime import datetime
import os
import json
import pandas as pd
from collections import deque

class OfflineDrowsinessDetector:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Eye and mouth landmark indices for MediaPipe
        self.LEFT_EYE_IDXS = [33, 7, 163, 144, 145, 153]
        self.RIGHT_EYE_IDXS = [362, 382, 381, 380, 374, 373]
        self.MOUTH_IDXS = [61, 84, 17, 314, 405, 320, 307, 375]
        
        # Detection thresholds
        self.EAR_THRESH = 0.3
        self.MAR_THRESH = 0.6
        self.CONSEC_FRAMES = 15
        self.YAWN_FRAMES = 10
        
        # Counters and tracking
        self.ear_counter = 0
        self.yawn_counter = 0
        self.total_blinks = 0
        self.total_yawns = 0
        self.drowsiness_events = 0
        
        # History for smoothing
        self.ear_history = deque(maxlen=30)
        self.mar_history = deque(maxlen=30)
        
        # Status flags
        self.drowsiness_detected = False
        self.yawn_detected = False
        self.alert_active = False
        
        # Initialize pygame for audio alerts
        try:
            pygame.mixer.init()
            self.create_alert_sound()
            self.audio_available = True
        except:
            self.audio_available = False
            
        # Session tracking
        self.session_data = []
        self.start_time = time.time()
        
    def create_alert_sound(self):
        """Create a simple beep sound for alerts"""
        try:
            sample_rate = 22050
            duration = 0.5
            frequency = 800
            
            frames = int(duration * sample_rate)
            arr = np.zeros(frames)
            
            for i in range(frames):
                arr[i] = np.sin(2 * np.pi * frequency * i / sample_rate)
            
            arr = (arr * 32767).astype(np.int16)
            sound = pygame.sndarray.make_sound(arr)
            self.alert_sound = sound
        except:
            self.alert_sound = None
            
    def calculate_ear(self, landmarks, w, h):
        """Calculate Eye Aspect Ratio"""
        try:
            points = []
            for landmark in landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                points.append([x, y])
            
            if len(points) >= 6:
                A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
                B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
                C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
                
                if C > 0:
                    ear = (A + B) / (2.0 * C)
                    return ear
            return 0.3
        except:
            return 0.3
            
    def calculate_mar(self, landmarks, w, h):
        """Calculate Mouth Aspect Ratio"""
        try:
            points = []
            for landmark in landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                points.append([x, y])
            
            if len(points) >= 6:
                A = np.linalg.norm(np.array(points[2]) - np.array(points[6]))
                B = np.linalg.norm(np.array(points[3]) - np.array(points[5]))
                C = np.linalg.norm(np.array(points[0]) - np.array(points[4]))
                
                if C > 0:
                    mar = (A + B) / (2.0 * C)
                    return mar
            return 0.3
        except:
            return 0.3
            
    def process_frame(self, frame):
        """Process frame for drowsiness detection"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        h, w, _ = frame.shape
        ear, mar = 0.3, 0.3
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract landmarks
                left_eye = [face_landmarks.landmark[i] for i in self.LEFT_EYE_IDXS]
                right_eye = [face_landmarks.landmark[i] for i in self.RIGHT_EYE_IDXS]
                mouth = [face_landmarks.landmark[i] for i in self.MOUTH_IDXS]
                
                # Calculate metrics
                left_ear = self.calculate_ear(left_eye, w, h)
                right_ear = self.calculate_ear(right_eye, w, h)
                ear = (left_ear + right_ear) / 2.0
                mar = self.calculate_mar(mouth, w, h)
                
                # Smooth values
                self.ear_history.append(ear)
                self.mar_history.append(mar)
                
                avg_ear = np.mean(list(self.ear_history)[-5:]) if len(self.ear_history) >= 5 else ear
                avg_mar = np.mean(list(self.mar_history)[-5:]) if len(self.mar_history) >= 5 else mar
                
                # Drowsiness detection
                if avg_ear < self.EAR_THRESH:
                    self.ear_counter += 1
                    if self.ear_counter >= self.CONSEC_FRAMES:
                        if not self.drowsiness_detected:
                            self.drowsiness_events += 1
                            self.trigger_alert("DROWSINESS")
                        self.drowsiness_detected = True
                else:
                    if self.ear_counter > 0 and self.drowsiness_detected:
                        self.total_blinks += 1
                    self.ear_counter = 0
                    self.drowsiness_detected = False
                
                # Yawn detection
                if avg_mar > self.MAR_THRESH:
                    self.yawn_counter += 1
                    if self.yawn_counter >= self.YAWN_FRAMES:
                        if not self.yawn_detected:
                            self.total_yawns += 1
                            self.trigger_alert("YAWN")
                        self.yawn_detected = True
                else:
                    self.yawn_counter = 0
                    self.yawn_detected = False
                
                # Draw landmarks and info
                self.draw_landmarks(frame, face_landmarks, w, h)
                self.draw_info(frame, avg_ear, avg_mar)
                
                # Log session data
                self.log_session_data(avg_ear, avg_mar)
                
        return frame, ear, mar
        
    def draw_landmarks(self, frame, face_landmarks, w, h):
        """Draw facial landmarks on frame"""
        # Draw eyes
        for idx in self.LEFT_EYE_IDXS + self.RIGHT_EYE_IDXS:
            x = int(face_landmarks.landmark[idx].x * w)
            y = int(face_landmarks.landmark[idx].y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
        # Draw mouth
        for idx in self.MOUTH_IDXS:
            x = int(face_landmarks.landmark[idx].x * w)
            y = int(face_landmarks.landmark[idx].y * h)
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            
    def draw_info(self, frame, ear, mar):
        """Draw information overlay on frame"""
        h, w, _ = frame.shape
        
        # Create background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Status text
        status = "ALERT!" if (self.drowsiness_detected or self.yawn_detected) else "AWAKE"
        color = (0, 0, 255) if (self.drowsiness_detected or self.yawn_detected) else (0, 255, 0)
        
        cv2.putText(frame, f"Status: {status}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"EAR: {ear:.3f}", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"MAR: {mar:.3f}", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Blinks: {self.total_blinks}", (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Yawns: {self.total_yawns}", (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Events: {self.drowsiness_events}", (15, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Alert messages
        if self.drowsiness_detected:
            cv2.putText(frame, "DROWSINESS DETECTED!", (15, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if self.yawn_detected:
            cv2.putText(frame, "YAWNING DETECTED!", (15, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            
    def trigger_alert(self, alert_type):
        """Trigger audio alert"""
        if not self.alert_active and self.audio_available and self.alert_sound:
            self.alert_active = True
            try:
                pygame.mixer.Sound.play(self.alert_sound)
            except:
                pass
            threading.Timer(1.0, self.reset_alert).start()
            
    def reset_alert(self):
        """Reset alert status"""
        self.alert_active = False
        
    def log_session_data(self, ear, mar):
        """Log session data"""
        timestamp = time.time() - self.start_time
        self.session_data.append({
            'timestamp': timestamp,
            'ear': ear,
            'mar': mar,
            'drowsy': self.drowsiness_detected,
            'yawn': self.yawn_detected
        })
        
    def save_session_data(self):
        """Save session data to files"""
        if self.session_data:
            os.makedirs("logs", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save detailed CSV
            df = pd.DataFrame(self.session_data)
            csv_filename = f"logs/session_{timestamp}.csv"
            df.to_csv(csv_filename, index=False)
            
            # Save summary JSON
            summary = {
                'timestamp': timestamp,
                'session_duration': time.time() - self.start_time,
                'total_blinks': self.total_blinks,
                'total_yawns': self.total_yawns,
                'drowsiness_events': self.drowsiness_events,
                'ear_threshold': self.EAR_THRESH,
                'mar_threshold': self.MAR_THRESH
            }
            
            json_filename = f"logs/session_{timestamp}.json"
            with open(json_filename, 'w') as f:
                json.dump(summary, f, indent=4)
            
            return csv_filename, json_filename
        return None, None

def main():
    st.set_page_config(page_title="Offline Drowsiness Detection", layout="wide")
    st.title("🚗 Offline Drowsiness Detection System")
    st.sidebar.title("Controls")
    
    # Initialize detector
    if 'detector' not in st.session_state:
        st.session_state.detector = OfflineDrowsinessDetector()
    
    detector = st.session_state.detector
    
    # Sidebar controls
    st.sidebar.subheader("Detection Thresholds")
    ear_thresh = st.sidebar.slider("EAR Threshold", 0.1, 0.4, 0.25, 0.01,key="ear_thresh_slider")
    mar_thresh = st.sidebar.slider("MAR Threshold", 0.3, 1.0, 0.6, 0.05,key="mar_thresh_slider")
    
    detector.EAR_THRESH = ear_thresh
    detector.MAR_THRESH = mar_thresh
    
    # Layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Camera Feed")
        camera_placeholder = st.empty()
        
    with col2:
        st.subheader("Real-time Metrics")
        metrics_placeholder = st.empty()
        
        st.subheader("Session Statistics")
        stats_placeholder = st.empty()
        
        if st.button("💾 Save Session Data"):
            csv_file, json_file = detector.save_session_data()
            if csv_file:
                st.success(f"Data saved!")
                st.text(f"CSV: {csv_file}")
                st.text(f"JSON: {json_file}")
            else:
                st.warning("No data to save")
                
        if st.button("🔄 Reset Session"):
            detector.total_blinks = 0
            detector.total_yawns = 0
            detector.drowsiness_events = 0
            detector.session_data = []
            detector.start_time = time.time()
            st.success("Session reset!")
    
    # Camera processing
    run = st.sidebar.checkbox("🎥 Start Detection", value=True,key="camera_check")
    
    if run:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            st.error("❌ Cannot open camera. Please check your webcam connection.")
            return
        
        try:
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to read from camera")
                    break
                    
                # Process frame
                processed_frame, ear, mar = detector.process_frame(frame)
                
                # Display frame
                camera_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
                
                # Update metrics
                with metrics_placeholder.container():
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Eye Aspect Ratio", f"{ear:.3f}", 
                                 delta=f"Threshold: {ear_thresh}")
                    with col_b:
                        st.metric("Mouth Aspect Ratio", f"{mar:.3f}", 
                                 delta=f"Threshold: {mar_thresh}")
                
                # Update stats
                session_time = time.time() - detector.start_time
                stats_text = f"""
                **Session Duration:** {session_time:.0f}s  
                **Blinks:** {detector.total_blinks}  
                **Yawns:** {detector.total_yawns}  
                **Drowsiness Events:** {detector.drowsiness_events}  
                **Status:** {'⚠️ ALERT' if (detector.drowsiness_detected or detector.yawn_detected) else '✅ AWAKE'}
                """
                stats_placeholder.markdown(stats_text)
                
                # Check if still running
               
                # Small delay
                time.sleep(0.03)
                
        finally:
            cap.release()
    else:
        st.info("Click 'Start Detection' to begin monitoring")

if __name__ == "__main__":
    main()
