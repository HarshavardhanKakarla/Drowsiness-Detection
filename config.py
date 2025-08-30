import json
import os
from datetime import datetime

class Config:
    """Configuration management for drowsiness detection system"""
    
    # Default Detection Parameters
    EAR_THRESHOLD = 0.3          # Eye Aspect Ratio threshold
    MAR_THRESHOLD = 0.6           # Mouth Aspect Ratio threshold  
    CONSECUTIVE_FRAMES = 15       # Frames for drowsiness confirmation
    YAWN_FRAMES = 10             # Frames for yawn confirmation
    
    # Camera Settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_BRIGHTNESS = 130
    CAMERA_INDEX = 0
    
    # Audio Settings
    ALERT_FREQUENCY = 800         # Hz
    ALERT_DURATION = 0.5         # seconds
    SAMPLE_RATE = 22050
    
    # Processing Settings
    SMOOTHING_WINDOW = 5         # Frames for smoothing
    MAX_HISTORY = 30            # Maximum history length
    
    # Logging Settings
    LOG_DIRECTORY = "logs"
    CONFIG_DIRECTORY = "config"
    DATA_DIRECTORY = "data"
    
    # MediaPipe Settings
    MIN_DETECTION_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.5
    MAX_NUM_FACES = 1
    
    @classmethod
    def save_config(cls, filename=None):
        """Save current configuration to JSON file"""
        if filename is None:
            os.makedirs(cls.CONFIG_DIRECTORY, exist_ok=True)
            filename = os.path.join(cls.CONFIG_DIRECTORY, "settings.json")
        
        config_dict = {
            'detection': {
                'ear_threshold': cls.EAR_THRESHOLD,
                'mar_threshold': cls.MAR_THRESHOLD,
                'consecutive_frames': cls.CONSECUTIVE_FRAMES,
                'yawn_frames': cls.YAWN_FRAMES,
                'smoothing_window': cls.SMOOTHING_WINDOW
            },
            'camera': {
                'width': cls.CAMERA_WIDTH,
                'height': cls.CAMERA_HEIGHT,
                'brightness': cls.CAMERA_BRIGHTNESS,
                'index': cls.CAMERA_INDEX
            },
            'audio': {
                'frequency': cls.ALERT_FREQUENCY,
                'duration': cls.ALERT_DURATION,
                'sample_rate': cls.SAMPLE_RATE
            },
            'mediapipe': {
                'min_detection_confidence': cls.MIN_DETECTION_CONFIDENCE,
                'min_tracking_confidence': cls.MIN_TRACKING_CONFIDENCE,
                'max_num_faces': cls.MAX_NUM_FACES
            },
            'metadata': {
                'saved_at': datetime.now().isoformat(),
                'version': '1.0.0'
            }
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(config_dict, f, indent=4)
            print(f"✅ Configuration saved to {filename}")
            return True
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")
            return False
    
    @classmethod
    def load_config(cls, filename=None):
        """Load configuration from JSON file"""
        if filename is None:
            filename = os.path.join(cls.CONFIG_DIRECTORY, "settings.json")
        
        try:
            with open(filename, 'r') as f:
                config_dict = json.load(f)
            
            # Load detection settings
            if 'detection' in config_dict:
                detection = config_dict['detection']
                cls.EAR_THRESHOLD = detection.get('ear_threshold', cls.EAR_THRESHOLD)
                cls.MAR_THRESHOLD = detection.get('mar_threshold', cls.MAR_THRESHOLD)
                cls.CONSECUTIVE_FRAMES = detection.get('consecutive_frames', cls.CONSECUTIVE_FRAMES)
                cls.YAWN_FRAMES = detection.get('yawn_frames', cls.YAWN_FRAMES)
                cls.SMOOTHING_WINDOW = detection.get('smoothing_window', cls.SMOOTHING_WINDOW)
            
            # Load camera settings
            if 'camera' in config_dict:
                camera = config_dict['camera']
                cls.CAMERA_WIDTH = camera.get('width', cls.CAMERA_WIDTH)
                cls.CAMERA_HEIGHT = camera.get('height', cls.CAMERA_HEIGHT)
                cls.CAMERA_BRIGHTNESS = camera.get('brightness', cls.CAMERA_BRIGHTNESS)
                cls.CAMERA_INDEX = camera.get('index', cls.CAMERA_INDEX)
            
            # Load audio settings
            if 'audio' in config_dict:
                audio = config_dict['audio']
                cls.ALERT_FREQUENCY = audio.get('frequency', cls.ALERT_FREQUENCY)
                cls.ALERT_DURATION = audio.get('duration', cls.ALERT_DURATION)
                cls.SAMPLE_RATE = audio.get('sample_rate', cls.SAMPLE_RATE)
            
            # Load MediaPipe settings
            if 'mediapipe' in config_dict:
                mediapipe = config_dict['mediapipe']
                cls.MIN_DETECTION_CONFIDENCE = mediapipe.get('min_detection_confidence', cls.MIN_DETECTION_CONFIDENCE)
                cls.MIN_TRACKING_CONFIDENCE = mediapipe.get('min_tracking_confidence', cls.MIN_TRACKING_CONFIDENCE)
                cls.MAX_NUM_FACES = mediapipe.get('max_num_faces', cls.MAX_NUM_FACES)
            
            print(f"✅ Configuration loaded from {filename}")
            return True
            
        except FileNotFoundError:
            print(f"⚠ Config file {filename} not found. Using default settings.")
            cls.save_config()  # Create default config file
            return False
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            return False
    
    @classmethod
    def reset_to_defaults(cls):
        """Reset all settings to default values"""
        cls.EAR_THRESHOLD = 0.3
        cls.MAR_THRESHOLD = 0.6
        cls.CONSECUTIVE_FRAMES = 15
        cls.YAWN_FRAMES = 10
        cls.CAMERA_WIDTH = 640
        cls.CAMERA_HEIGHT = 480
        cls.ALERT_FREQUENCY = 800
        cls.ALERT_DURATION = 0.5
        cls.SAMPLE_RATE = 22050
        print("✅ Configuration reset to defaults")
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        directories = [cls.LOG_DIRECTORY, cls.CONFIG_DIRECTORY, cls.DATA_DIRECTORY]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        print("✅ Directories created")
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("\n📋 Current Configuration:")
        print("=" * 40)
        print(f"EAR Threshold: {cls.EAR_THRESHOLD}")
        print(f"MAR Threshold: {cls.MAR_THRESHOLD}")
        print(f"Consecutive Frames: {cls.CONSECUTIVE_FRAMES}")
        print(f"Yawn Frames: {cls.YAWN_FRAMES}")
        print(f"Camera Resolution: {cls.CAMERA_WIDTH}x{cls.CAMERA_HEIGHT}")
        print(f"Alert Frequency: {cls.ALERT_FREQUENCY} Hz")
        print(f"Alert Duration: {cls.ALERT_DURATION}s")
        print("=" * 40)

# Initialize default configuration on import
if __name__ == "__main__":
    Config.create_directories()
    Config.save_config()
    Config.print_config()
else:
    # Auto-create directories and load config when imported
    Config.create_directories()
    Config.load_config()
