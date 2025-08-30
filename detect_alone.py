import cv2
import numpy as np
import mediapipe as mp
import pygame
import time
import threading
import os
import json
from datetime import datetime
from collections import deque

class StandaloneDrowsinessDetector:
    def __init__(self):
        print("Initializing Offline Drowsiness Detection System...")
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Eye and mouth landmark indices
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
        
        # Initialize pygame for audio
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.create_alert_sound()
            self.audio_available = True
            print("✓ Audio system initialized")
        except:
            print("⚠ Audio system not available")
            self.audio_available = False
            
        # Session tracking
        self.session_start = time.time()
        self.session_data = []
        
        print("✓ System initialized successfully!")
        
    def create_alert_sound(self):
        """Create alert sound"""
        duration = 0.5
        sample_rate = 22050
        frequency = 800
        
        frames = int(duration * sample_rate)
        arr = np.zeros(frames)
        
        for i in range(frames):
            arr[i] = np.sin(2 * np.pi * frequency * i / sample_rate)
        
        arr = (arr * 32767).astype(np.int16)
        stereo_arr = np.zeros((frames, 2), dtype=np.int16)
        stereo_arr[:, 0] = arr
        stereo_arr[:, 1] = arr
        
        self.alert_sound = pygame.sndarray.make_sound(stereo_arr)
        
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
        """Process single frame for drowsiness detection"""
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
                
                # Draw landmarks
                self.draw_landmarks(frame, face_landmarks, w, h)
                
                # Log data
                self.log_session_data(avg_ear, avg_mar)
                
        # Draw information overlay
        self.draw_info(frame, ear, mar)
        
        return frame, ear, mar
        
    def draw_landmarks(self, frame, face_landmarks, w, h):
        """Draw facial landmarks"""
        # Draw eyes (green)
        for idx in self.LEFT_EYE_IDXS + self.RIGHT_EYE_IDXS:
            x = int(face_landmarks.landmark[idx].x * w)
            y = int(face_landmarks.landmark[idx].y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
        # Draw mouth (red)
        for idx in self.MOUTH_IDXS:
            x = int(face_landmarks.landmark[idx].x * w)
            y = int(face_landmarks.landmark[idx].y * h)
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            
    def draw_info(self, frame, ear, mar):
        """Draw information overlay"""
        h, w, _ = frame.shape
        
        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 220), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Status
        status = "ALERT!" if (self.drowsiness_detected or self.yawn_detected) else "AWAKE"
        color = (0, 0, 255) if (self.drowsiness_detected or self.yawn_detected) else (0, 255, 0)
        
        cv2.putText(frame, f"Status: {status}", (15, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Metrics
        cv2.putText(frame, f"EAR: {ear:.3f} (Thresh: {self.EAR_THRESH})", (15, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"MAR: {mar:.3f} (Thresh: {self.MAR_THRESH})", (15, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Counters
        cv2.putText(frame, f"Blinks: {self.total_blinks}", (15, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Yawns: {self.total_yawns}", (15, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Drowsiness Events: {self.drowsiness_events}", (15, 165),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Session time
        session_time = time.time() - self.session_start
        cv2.putText(frame, f"Session: {session_time:.0f}s", (15, 190),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Alert messages
        if self.drowsiness_detected:
            cv2.putText(frame, "DROWSINESS DETECTED!", (15, h-60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        if self.yawn_detected:
            cv2.putText(frame, "YAWNING DETECTED!", (15, h-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
                       
        # Instructions
        cv2.putText(frame, "Press 'q' to quit, 's' to save session", 
                   (w-350, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
    def trigger_alert(self, alert_type):
        """Trigger audio alert"""
        if not self.alert_active and self.audio_available:
            self.alert_active = True
            try:
                pygame.mixer.Sound.play(self.alert_sound)
                print(f"🚨 {alert_type} ALERT!")
            except:
                pass
            threading.Timer(1.0, self.reset_alert).start()
            
    def reset_alert(self):
        """Reset alert status"""
        self.alert_active = False
        
    def log_session_data(self, ear, mar):
        """Log session data"""
        timestamp = time.time() - self.session_start
        self.session_data.append({
            'timestamp': timestamp,
            'ear': ear,
            'mar': mar,
            'drowsy': self.drowsiness_detected,
            'yawn': self.yawn_detected
        })
        
    def save_session_data(self):
        """Save session statistics"""
        session_time = time.time() - self.session_start
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        
        # Save detailed session data as CSV
        if self.session_data:
            import pandas as pd
            df = pd.DataFrame(self.session_data)
            csv_filename = f"logs/session_{timestamp}.csv"
            df.to_csv(csv_filename, index=False)
        
        # Save session summary as JSON
        session_summary = {
            'timestamp': timestamp,
            'session_duration': session_time,
            'total_blinks': self.total_blinks,
            'total_yawns': self.total_yawns,
            'drowsiness_events': self.drowsiness_events,
            'ear_threshold': self.EAR_THRESH,
            'mar_threshold': self.MAR_THRESH
        }
        
        json_filename = f"logs/session_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(session_summary, f, indent=4)
            
        return json_filename
        
    def run(self):
        """Main execution loop"""
        print("\n🚗 Starting Drowsiness Detection System...")
        print("📋 Controls:")
        print("   - Press 'q' to quit")
        print("   - Press 's' to save session data")
        print("   - Press 'r' to reset session")
        print()
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("❌ Error: Cannot open camera")
            return
            
        print("✅ Camera initialized successfully")
        print("🎥 Detection started... (Press 'q' to quit)")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Cannot read from camera")
                    break
                    
                # Process frame
                processed_frame, ear, mar = self.process_frame(frame)
                
                # Display frame
                cv2.imshow("Offline Drowsiness Detection System", processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n🛑 Stopping detection system...")
                    break
                elif key == ord('s'):
                    filename = self.save_session_data()
                    print(f"💾 Session data saved to {filename}")
                elif key == ord('r'):
                    print("🔄 Resetting session...")
                    self.total_blinks = 0
                    self.total_yawns = 0
                    self.drowsiness_events = 0
                    self.session_data = []
                    self.session_start = time.time()
                    
        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user")
            
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            if self.audio_available:
                pygame.mixer.quit()
            
            # Save final session data
            filename = self.save_session_data()
            print(f"💾 Final session data saved to {filename}")
            
            # Print session summary
            session_time = time.time() - self.session_start
            print("\n📊 Session Summary:")
            print(f"   Duration: {session_time:.1f} seconds")
            print(f"   Total Blinks: {self.total_blinks}")
            print(f"   Total Yawns: {self.total_yawns}")
            print(f"   Drowsiness Events: {self.drowsiness_events}")
            print("\n✅ System shutdown complete")

def main():
    try:
        detector = StandaloneDrowsinessDetector()
        detector.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please ensure you have:")
        print("  - A working webcam")
        print("  - All required packages installed")
        print("  - Proper permissions for camera access")

if __name__ == "__main__":
    main()
