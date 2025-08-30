#!/usr/bin/env python3
"""
Setup script for Offline Drowsiness Detection System
Handles installation, dependency checking, and initial configuration
"""

import os
import sys
import subprocess
import platform
import importlib.util

def print_header():
    """Print setup header"""
    print("=" * 60)
    print("🚗 OFFLINE DROWSINESS DETECTION SYSTEM SETUP")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 7):
        print(f"❌ Python 3.7+ required. Current version: {sys.version.split()[0]}")
        return False
    else:
        print(f"✅ Python {sys.version.split()[0]} detected")
        return True

def check_system_info():
    """Display system information"""
    print("\n💻 System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Architecture: {platform.machine()}")
    print(f"   Processor: {platform.processor()}")

def install_requirements():
    """Install required packages from requirements.txt"""
    print("\n📦 Installing required packages...")
    
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found")
        return False
    
    try:
        # Upgrade pip first
        print("   📈 Upgrading pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Install requirements
        print("   📦 Installing packages...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All packages installed successfully")
            return True
        else:
            print(f"❌ Error installing packages: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during installation: {e}")
        return False

def create_directories():
    """Create necessary project directories"""
    print("\n📁 Creating project directories...")
    
    directories = [
        'logs',           # Session data logs
        'config',         # Configuration files
        'data',           # Sample data and resources
        'utils',          # Utility modules
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"   ✅ {directory}/")
        except Exception as e:
            print(f"   ❌ Failed to create {directory}/: {e}")
            return False
    
    return True

def test_camera():
    """Test camera availability"""
    print("\n📷 Testing camera access...")
    
    try:
        import cv2
        
        # Try to open camera
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
                print(f"✅ Camera detected: {width}x{height}")
                cap.release()
                return True
            else:
                print("❌ Camera detected but cannot read frames")
                cap.release()
                return False
        else:
            print("❌ Cannot access camera (index 0)")
            
            # Try other camera indices
            for i in range(1, 4):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        print(f"✅ Camera found at index {i}")
                        cap.release()
                        return True
                    cap.release()
            
            print("❌ No working camera found")
            return False
            
    except ImportError:
        print("❌ OpenCV not available for camera testing")
        return False
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_audio():
    """Test audio system"""
    print("\n🔊 Testing audio system...")
    
    try:
        import pygame
        pygame.mixer.init()
        print("✅ Audio system available")
        pygame.mixer.quit()
        return True
    except ImportError:
        print("⚠ Pygame not available - audio alerts disabled")
        return False
    except Exception as e:
        print(f"⚠ Audio system may not work: {e}")
        return False

def test_mediapipe():
    """Test MediaPipe functionality"""
    print("\n🤖 Testing MediaPipe...")
    
    try:
        import mediapipe as mp
        
        # Test face mesh initialization
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
        
        print("✅ MediaPipe face detection available")
        return True
        
    except ImportError:
        print("❌ MediaPipe not available")
        return False
    except Exception as e:
        print(f"❌ MediaPipe test failed: {e}")
        return False

def create_config():
    """Create initial configuration"""
    print("\n⚙️ Creating initial configuration...")
    
    try:
        from config import Config
        Config.create_directories()
        Config.save_config()
        print("✅ Default configuration created")
        return True
    except Exception as e:
        print(f"❌ Configuration creation failed: {e}")
        return False

def run_basic_test():
    """Run basic functionality test"""
    print("\n🧪 Running basic functionality test...")
    
    try:
        # Test imports
        import cv2
        import mediapipe as mp
        import numpy as np
        
        # Create a simple test
        print("   📹 Testing OpenCV...")
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        print("   🤖 Testing MediaPipe...")
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
        
        print("✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False

def print_usage_instructions():
    """Print usage instructions"""
    print("\n" + "=" * 60)
    print("🎯 SETUP COMPLETE - USAGE INSTRUCTIONS")
    print("=" * 60)
    print()
    print("📋 Available Applications:")
    print("   1. 🌐 Streamlit Web Interface:")
    print("      streamlit run drowsiness_detection_offline.py")
    print()
    print("   2. 🖥️ Standalone OpenCV Application:")
    print("      python drowsiness_detection_standalone.py")
    print()
    print("   3. 📊 Data Analysis Tools:")
    print("      python utils/analysis.py")
    print()
    print("⚙️ Configuration:")
    print("   • Edit config.py to adjust detection parameters")
    print("   • Modify thresholds in config/settings.json")
    print("   • Check logs/ directory for session data")
    print()
    print("🆘 Troubleshooting:")
    print("   • Ensure camera permissions are granted")
    print("   • Try different camera indices if camera fails")
    print("   • Check logs/ directory for error information")
    print()

def print_summary(results):
    """Print setup summary"""
    print("\n" + "=" * 60)
    print("📋 SETUP SUMMARY")
    print("=" * 60)
    
    components = [
        ("Python Version", results['python']),
        ("Dependencies", results['packages']),
        ("Directories", results['directories']),
        ("Camera", results['camera']),
        ("Audio", results['audio']),
        ("MediaPipe", results['mediapipe']),
        ("Configuration", results['config']),
        ("Basic Test", results['basic_test'])
    ]
    
    for component, status in components:
        icon = "✅" if status else "❌"
        print(f"   {icon} {component}")
    
    # Overall status
    critical_components = ['python', 'packages', 'directories', 'mediapipe', 'config']
    critical_passed = all(results[comp] for comp in critical_components)
    
    print()
    if critical_passed:
        print("🎉 SETUP SUCCESSFUL!")
        print("   System is ready for drowsiness detection")
        if not results['camera']:
            print("   ⚠ Camera issues detected - please resolve before use")
    else:
        print("❌ SETUP INCOMPLETE")
        print("   Please resolve the failed components above")

def main():
    """Main setup function"""
    print_header()
    
    # Track setup results
    results = {
        'python': False,
        'packages': False,
        'directories': False,
        'camera': False,
        'audio': False,
        'mediapipe': False,
        'config': False,
        'basic_test': False
    }
    
    # Run setup steps
    try:
        results['python'] = check_python_version()
        if not results['python']:
            print("\n❌ Setup cannot continue with incompatible Python version")
            return
        
        check_system_info()
        results['directories'] = create_directories()
        results['packages'] = install_requirements()
        
        if results['packages']:
            results['camera'] = test_camera()
            results['audio'] = test_audio()
            results['mediapipe'] = test_mediapipe()
            results['config'] = create_config()
            results['basic_test'] = run_basic_test()
        
        # Print results
        print_summary(results)
        
        # Show usage instructions if setup was successful
        critical_components = ['python', 'packages', 'directories', 'mediapipe', 'config']
        if all(results[comp] for comp in critical_components):
            print_usage_instructions()
        
    except KeyboardInterrupt:
        print("\n\n⚠ Setup interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error during setup: {e}")

if __name__ == "__main__":
    main()
