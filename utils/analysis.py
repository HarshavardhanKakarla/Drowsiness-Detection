import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from glob import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SessionAnalyzer:
    """Comprehensive analysis tools for drowsiness detection sessions"""
    
    def __init__(self, logs_directory="logs"):
        self.logs_dir = logs_directory
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir, exist_ok=True)
            print(f"📁 Created logs directory: {self.logs_dir}")
        
    def load_session_json(self, filename):
        """Load session summary from JSON file"""
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return None
            
    def load_session_csv(self, filename):
        """Load detailed session data from CSV file"""
        try:
            return pd.read_csv(filename)
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return None
    
    def get_all_sessions(self):
        """Get all session files"""
        json_files = glob(os.path.join(self.logs_dir, "*.json"))
        csv_files = glob(os.path.join(self.logs_dir, "*.csv"))
        
        print(f"📊 Found {len(json_files)} session summaries and {len(csv_files)} detailed logs")
        
        return json_files, csv_files
    
    def analyze_session_summaries(self):
        """Analyze all session summary data"""
        json_files, _ = self.get_all_sessions()
        
        if not json_files:
            print("⚠ No session summary files found")
            return None
            
        # Load all session summaries
        sessions = []
        for json_file in json_files:
            data = self.load_session_json(json_file)
            if data:
                sessions.append(data)
                
        if not sessions:
            print("⚠ No valid session data found")
            return None
            
        df = pd.DataFrame(sessions)
        
        # Create visualization
        self.plot_session_summaries(df)
        
        # Print statistics
        self.print_summary_statistics(df)
        
        return df
    
    def plot_session_summaries(self, df):
        """Create comprehensive visualization of session summaries"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Drowsiness Detection Session Analysis', fontsize=16, fontweight='bold')
        
        # 1. Session Duration Distribution
        axes[0, 0].hist(df['session_duration'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Session Duration Distribution')
        axes[0, 0].set_xlabel('Duration (seconds)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Blinks vs Session Duration
        axes[0, 1].scatter(df['session_duration'], df['total_blinks'], alpha=0.6, color='green')
        axes[0, 1].set_title('Blinks vs Session Duration')
        axes[0, 1].set_xlabel('Duration (seconds)')
        axes[0, 1].set_ylabel('Total Blinks')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Yawns vs Session Duration
        axes[0, 2].scatter(df['session_duration'], df['total_yawns'], alpha=0.6, color='orange')
        axes[0, 2].set_title('Yawns vs Session Duration')
        axes[0, 2].set_xlabel('Duration (seconds)')
        axes[0, 2].set_ylabel('Total Yawns')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Drowsiness Events Distribution
        axes[1, 0].hist(df['drowsiness_events'], bins=max(1, len(df)//3), alpha=0.7, color='red', edgecolor='black')
        axes[1, 0].set_title('Drowsiness Events Distribution')
        axes[1, 0].set_xlabel('Events Count')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. EAR Threshold Usage
        if 'ear_threshold' in df.columns:
            ear_counts = df['ear_threshold'].value_counts()
            axes[1, 1].pie(ear_counts.values, labels=ear_counts.index, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('EAR Threshold Usage')
        else:
            axes[1, 1].text(0.5, 0.5, 'No EAR threshold data', ha='center', va='center')
            axes[1, 1].set_title('EAR Threshold Usage')
        
        # 6. Performance Metrics
        avg_blinks_per_min = (df['total_blinks'] / (df['session_duration'] / 60)).mean()
        avg_yawns_per_session = df['total_yawns'].mean()
        total_drowsiness_events = df['drowsiness_events'].sum()
        
        metrics_text = f"""
        Performance Metrics:
        
        Avg Blinks/min: {avg_blinks_per_min:.1f}
        Avg Yawns/session: {avg_yawns_per_session:.1f}
        Total Drowsiness Events: {total_drowsiness_events}
        Sessions with Events: {(df['drowsiness_events'] > 0).sum()}
        """
        
        axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center')
        axes[1, 2].axis('off')
        axes[1, 2].set_title('Summary Metrics')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = os.path.join(self.logs_dir, 'session_analysis.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📊 Analysis plot saved to {plot_filename}")
        
        plt.show()
    
    def analyze_detailed_session(self, csv_filename):
        """Analyze detailed CSV session data"""
        df = self.load_session_csv(csv_filename)
        if df is None or df.empty:
            print("⚠ No detailed session data found")
            return
            
        print(f"\n📈 Analyzing detailed session: {os.path.basename(csv_filename)}")
        
        # Create detailed visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Detailed Session Analysis: {os.path.basename(csv_filename)}', fontsize=14)
        
        # 1. EAR over time
        axes[0, 0].plot(df['timestamp'], df['ear'], label='EAR', alpha=0.7, color='blue')
        axes[0, 0].axhline(y=0.25, color='r', linestyle='--', label='Threshold (0.25)')
        axes[0, 0].fill_between(df['timestamp'], df['ear'], alpha=0.3, color='blue')
        axes[0, 0].set_title('Eye Aspect Ratio Over Time')
        axes[0, 0].set_xlabel('Time (seconds)')
        axes[0, 0].set_ylabel('EAR')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. MAR over time
        axes[0, 1].plot(df['timestamp'], df['mar'], label='MAR', color='orange', alpha=0.7)
        axes[0, 1].axhline(y=0.6, color='r', linestyle='--', label='Threshold (0.6)')
        axes[0, 1].fill_between(df['timestamp'], df['mar'], alpha=0.3, color='orange')
        axes[0, 1].set_title('Mouth Aspect Ratio Over Time')
        axes[0, 1].set_xlabel('Time (seconds)')
        axes[0, 1].set_ylabel('MAR')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Events over time
        drowsy_times = df[df['drowsy'] == True]['timestamp'] if 'drowsy' in df.columns else []
        yawn_times = df[df['yawn'] == True]['timestamp'] if 'yawn' in df.columns else []
        
        if len(drowsy_times) > 0 or len(yawn_times) > 0:
            if len(drowsy_times) > 0:
                axes[1, 0].scatter(drowsy_times, [1]*len(drowsy_times), color='red', label='Drowsiness', alpha=0.7, s=50)
            if len(yawn_times) > 0:
                axes[1, 0].scatter(yawn_times, [0]*len(yawn_times), color='blue', label='Yawns', alpha=0.7, s=50)
            axes[1, 0].set_title('Detection Events Over Time')
            axes[1, 0].set_xlabel('Time (seconds)')
            axes[1, 0].set_ylabel('Event Type')
            axes[1, 0].set_yticks([0, 1])
            axes[1, 0].set_yticklabels(['Yawn', 'Drowsiness'])
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No events detected', ha='center', va='center')
            axes[1, 0].set_title('Detection Events Over Time')
        
        # 4. Statistics
        stats_text = f"""
        Session Statistics:
        
        Duration: {df['timestamp'].max():.1f}s
        Avg EAR: {df['ear'].mean():.3f}
        Min EAR: {df['ear'].min():.3f}
        Avg MAR: {df['mar'].mean():.3f}
        Max MAR: {df['mar'].max():.3f}
        
        Drowsy Frames: {df['drowsy'].sum() if 'drowsy' in df.columns else 0}
        Yawn Frames: {df['yawn'].sum() if 'yawn' in df.columns else 0}
        
        EAR Below Threshold: {(df['ear'] < 0.25).sum()}
        MAR Above Threshold: {(df['mar'] > 0.6).sum()}
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center')
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Session Statistics')
        
        plt.tight_layout()
        
        # Save detailed plot
        base_name = os.path.splitext(os.path.basename(csv_filename))[0]
        plot_filename = os.path.join(self.logs_dir, f'{base_name}_detailed_analysis.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📊 Detailed analysis plot saved to {plot_filename}")
        
        plt.show()
    
    def print_summary_statistics(self, df):
        """Print comprehensive summary statistics"""
        print("\n" + "="*60)
        print("📊 SESSION ANALYSIS SUMMARY")
        print("="*60)
        
        total_sessions = len(df)
        total_duration = df['session_duration'].sum()
        avg_duration = df['session_duration'].mean()
        total_blinks = df['total_blinks'].sum()
        total_yawns = df['total_yawns'].sum()
        total_events = df['drowsiness_events'].sum()
        
        print(f"📈 Overall Statistics:")
        print(f"   Total Sessions: {total_sessions}")
        print(f"   Total Duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
        print(f"   Average Session: {avg_duration:.1f} seconds")
        print(f"   Total Blinks: {total_blinks}")
        print(f"   Total Yawns: {total_yawns}")
        print(f"   Total Drowsiness Events: {total_events}")
        
        print(f"\n📊 Performance Metrics:")
        print(f"   Avg Blinks per Minute: {(total_blinks / (total_duration / 60)):.1f}")
        print(f"   Avg Yawns per Session: {total_yawns / total_sessions:.1f}")
        print(f"   Sessions with Drowsiness: {(df['drowsiness_events'] > 0).sum()} ({(df['drowsiness_events'] > 0).mean()*100:.1f}%)")
        
        if 'ear_threshold' in df.columns:
            print(f"\n⚙️ Configuration Usage:")
            print(f"   Most Common EAR Threshold: {df['ear_threshold'].mode().iloc[0]}")
            print(f"   Most Common MAR Threshold: {df['mar_threshold'].mode().iloc[0]}")
        
        print("="*60)
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report"""
        print("\n🔍 Generating Comprehensive Analysis Report...")
        
        json_files, csv_files = self.get_all_sessions()
        
        if not json_files and not csv_files:
            print("⚠ No session data found to analyze")
            return
        
        report_lines = []
        report_lines.append("DROWSINESS DETECTION ANALYSIS REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Analyze summaries
        if json_files:
            df_summary = self.analyze_session_summaries()
            if df_summary is not None:
                report_lines.append("SESSION SUMMARY ANALYSIS")
                report_lines.append("-" * 30)
                report_lines.append(f"Total Sessions Analyzed: {len(df_summary)}")
                report_lines.append(f"Total Duration: {df_summary['session_duration'].sum():.1f} seconds")
                report_lines.append(f"Average Session Duration: {df_summary['session_duration'].mean():.1f} seconds")
                report_lines.append(f"Total Blinks: {df_summary['total_blinks'].sum()}")
                report_lines.append(f"Total Yawns: {df_summary['total_yawns'].sum()}")
                report_lines.append(f"Total Drowsiness Events: {df_summary['drowsiness_events'].sum()}")
                report_lines.append("")
        
        # Analyze detailed sessions
        if csv_files:
            report_lines.append("DETAILED SESSION ANALYSIS")
            report_lines.append("-" * 30)
            report_lines.append(f"Detailed Sessions Available: {len(csv_files)}")
            
            for csv_file in csv_files[-3:]:  # Analyze last 3 sessions
                df_detail = self.load_session_csv(csv_file)
                if df_detail is not None:
                    filename = os.path.basename(csv_file)
                    report_lines.append(f"\nSession: {filename}")
                    report_lines.append(f"  Duration: {df_detail['timestamp'].max():.1f}s")
                    report_lines.append(f"  Avg EAR: {df_detail['ear'].mean():.3f}")
                    report_lines.append(f"  Avg MAR: {df_detail['mar'].mean():.3f}")
                    if 'drowsy' in df_detail.columns:
                        report_lines.append(f"  Drowsy Frames: {df_detail['drowsy'].sum()}")
                    if 'yawn' in df_detail.columns:
                        report_lines.append(f"  Yawn Frames: {df_detail['yawn'].sum()}")
        
        # Save report
        report_text = "\n".join(report_lines)
        report_filename = os.path.join(self.logs_dir, 'comprehensive_analysis_report.txt')
        
        with open(report_filename, 'w') as f:
            f.write(report_text)
        
        print(f"📄 Comprehensive report saved to {report_filename}")
        print("\n" + report_text)
        
        return report_filename

def main():
    """Main function for interactive analysis"""
    print("🔍 Drowsiness Detection Session Analyzer")
    print("=" * 45)
    
    analyzer = SessionAnalyzer()
    
    json_files, csv_files = analyzer.get_all_sessions()
    
    if not json_files and not csv_files:
        print("⚠ No session data found.")
        print("💡 Run some detection sessions first to generate data for analysis.")
        return
    
    while True:
        print("\n📋 Analysis Options:")
        print("1. 📊 Analyze All Session Summaries")
        print("2. 📈 Analyze Specific Detailed Session")
        print("3. 📄 Generate Comprehensive Report")
        print("4. 🚪 Exit")
        
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                analyzer.analyze_session_summaries()
                
            elif choice == '2':
                if csv_files:
                    print("\n📁 Available detailed sessions:")
                    for i, file in enumerate(csv_files):
                        print(f"   {i+1}. {os.path.basename(file)}")
                    
                    try:
                        file_choice = int(input("Select file number: ")) - 1
                        if 0 <= file_choice < len(csv_files):
                            analyzer.analyze_detailed_session(csv_files[file_choice])
                        else:
                            print("❌ Invalid file selection")
                    except ValueError:
                        print("❌ Please enter a valid number")
                else:
                    print("⚠ No detailed session files found")
                    
            elif choice == '3':
                analyzer.generate_comprehensive_report()
                
            elif choice == '4':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please select 1-4.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
