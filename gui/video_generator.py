#!/usr/bin/env python3
"""
Video Generator for Molecular Visualizer
Creates placeholder videos or helps download/generate molecular animation videos
"""
import os
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QProgressBar, QTextEdit
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette
import requests
import subprocess

class VideoGeneratorWidget(QWidget):
    """Widget for generating or downloading molecular videos."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Molecular Video Generator")
        self.setGeometry(200, 200, 800, 600)
        
        # Dark theme
        self.setStyleSheet("""
            QWidget {
                background-color: rgb(20, 15, 30);
                color: white;
            }
            QPushButton {
                background-color: rgb(60, 40, 100);
                border: 2px solid rgb(100, 70, 150);
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgb(80, 60, 120);
                border-color: rgb(120, 90, 170);
            }
            QPushButton:pressed {
                background-color: rgb(50, 30, 90);
            }
            QLabel {
                font-size: 14px;
                padding: 5px;
            }
            QTextEdit {
                background-color: rgb(30, 25, 40);
                border: 1px solid rgb(80, 60, 120);
                border-radius: 5px;
                padding: 10px;
                font-family: 'Courier New';
            }
            QProgressBar {
                border: 2px solid rgb(100, 70, 150);
                border-radius: 5px;
                text-align: center;
                background-color: rgb(30, 25, 40);
            }
            QProgressBar::chunk {
                background-color: rgb(120, 80, 200);
                border-radius: 3px;
            }
        """)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Molecular Video Generator")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: rgb(200, 150, 255); padding: 20px;")
        layout.addWidget(title)
        
        # Instructions
        instructions = QLabel("""
This tool helps you create videos for the molecular visualizer.
Choose one of the options below to generate molecular animation videos.
        """)
        instructions.setStyleSheet("color: rgb(180, 140, 220); padding: 10px;")
        layout.addWidget(instructions)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.create_placeholder_btn = QPushButton("Create Placeholder Videos")
        self.create_placeholder_btn.clicked.connect(self.create_placeholder_videos)
        button_layout.addWidget(self.create_placeholder_btn)
        
        self.download_videos_btn = QPushButton("Download Sample Videos")
        self.download_videos_btn.clicked.connect(self.download_sample_videos)
        button_layout.addWidget(self.download_videos_btn)
        
        self.blender_help_btn = QPushButton("Blender Instructions")
        self.blender_help_btn.clicked.connect(self.show_blender_instructions)
        button_layout.addWidget(self.blender_help_btn)
        
        layout.addLayout(button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Log output
        self.log_output = QTextEdit()
        self.log_output.setMaximumHeight(200)
        self.log_output.setPlainText("Ready to generate videos...\n")
        layout.addWidget(self.log_output)
        
        self.setLayout(layout)
    
    def log_message(self, message):
        """Add a message to the log output."""
        self.log_output.append(message)
        self.log_output.repaint()
    
    def create_placeholder_videos(self):
        """Create simple placeholder videos using FFmpeg."""
        self.log_message("Creating placeholder videos...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        molecules = ["h2", "nh3", "ch4", "h2o", "co2", "c2h6"]
        colors = {
            "h2": "white",
            "nh3": "blue", 
            "ch4": "gray",
            "h2o": "red",
            "co2": "green",
            "c2h6": "yellow"
        }
        
        videos_dir = os.path.join("gui", "videos")
        os.makedirs(videos_dir, exist_ok=True)
        
        total_molecules = len(molecules)
        
        for i, molecule in enumerate(molecules):
            try:
                self.log_message(f"Creating {molecule} placeholder...")
                
                # Create a simple rotating colored circle video using FFmpeg
                output_path = os.path.join(videos_dir, f"{molecule}_rotation.mp4")
                color = colors.get(molecule, "white")
                
                # FFmpeg command to create a rotating colored shape
                cmd = [
                    "ffmpeg", "-y",  # -y to overwrite
                    "-f", "lavfi",
                    "-i", f"color=c=black:size=1280x720:duration=10:rate=30",
                    "-f", "lavfi", 
                    "-i", f"color=c={color}:size=100x100:duration=10:rate=30",
                    "-filter_complex", f"[1]rotate=2*PI*t/10:fillcolor=none:ow=1280:oh=720[rot];[0][rot]overlay=(W-w)/2:(H-h)/2",
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    output_path
                ]
                
                # Try to run FFmpeg
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.log_message(f"✓ Created {molecule}_rotation.mp4")
                else:
                    self.log_message(f"⚠ FFmpeg not available, creating simple file for {molecule}")
                    # Create an empty file as placeholder
                    with open(output_path, 'w') as f:
                        f.write(f"# Placeholder for {molecule} video\n")
                
                progress = int((i + 1) / total_molecules * 100)
                self.progress_bar.setValue(progress)
                
            except Exception as e:
                self.log_message(f"✗ Error creating {molecule}: {str(e)}")
        
        self.log_message("Placeholder video creation completed!")
        self.progress_bar.setVisible(False)
    
    def download_sample_videos(self):
        """Attempt to download sample molecular videos."""
        self.log_message("Searching for sample molecular videos...")
        self.log_message("Note: You'll need to manually add videos to gui/videos/")
        self.log_message("Recommended sources:")
        self.log_message("- https://www.rcsb.org/ (Protein Data Bank)")
        self.log_message("- ChemSketch molecular viewer exports")
        self.log_message("- Blender molecular animation renders")
        self.log_message("- Screen recordings from online molecular viewers")
    
    def show_blender_instructions(self):
        """Show instructions for creating videos with Blender."""
        instructions = """
BLENDER MOLECULAR VIDEO CREATION:

1. Install Blender (free from blender.org)
2. Install molecular add-ons:
   - Atomic Blender (PDB import)
   - Chemistry add-on
   
3. Create molecular videos:
   a) Import molecule (File > Import > PDB)
   b) Set up materials and lighting
   c) Create rotation animation:
      - Select molecule
      - Insert keyframe at frame 1 (rotation 0°)
      - Go to frame 300 
      - Rotate molecule 360° on Z-axis
      - Insert keyframe
   d) Render animation:
      - Set output to MP4
      - Render > Render Animation
      
4. Export as {molecule}_rotation.mp4 to gui/videos/

NAMING CONVENTION:
- h2_rotation.mp4
- nh3_rotation.mp4
- ch4_rotation.mp4
- h2o_rotation.mp4
- co2_rotation.mp4
- c2h6_rotation.mp4
        """
        
        self.log_output.setPlainText(instructions)

def main():
    """Main function to run the video generator."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(20, 15, 30))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    generator = VideoGeneratorWidget()
    generator.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
