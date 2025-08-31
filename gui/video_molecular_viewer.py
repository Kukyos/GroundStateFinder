#!/usr/bin/env python3
"""
Video-based Molecular Visualizer
Uses pre-recorded video renders of 3D molecular models as backgrounds
This provides high-quality molecular visualization without OpenGL complexity
"""
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                             QComboBox, QHBoxLayout, QLabel)
from PyQt5.QtCore import QTimer, Qt, QUrl
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget

print("[VideoMol] Starting video-based molecular visualizer...")

class VideoMolecularViewer(QVideoWidget):
    """Video widget for displaying pre-recorded molecular animations."""
    
    def __init__(self):
        super().__init__()
        
        # Set up media player
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self)
        
        # Current molecule
        self.current_molecule = "H2"
        
        # Set widget properties
        self.setMinimumSize(600, 600)
        self.setStyleSheet("""
            QVideoWidget {
                background-color: rgb(12, 8, 20);
                border: 2px solid rgb(80, 40, 120);
                border-radius: 10px;
            }
        """)
        
        # Media player signals
        self.media_player.mediaStatusChanged.connect(self.on_media_status_changed)
        self.media_player.error.connect(self.on_media_error)
        
        print("[VideoMol] Video widget created")
    
    def load_molecule_video(self, molecule_name):
        """Load and play a video for the specified molecule."""
        self.current_molecule = molecule_name
        
        # Video file path (we'll create these)
        video_path = os.path.join("gui", "videos", f"{molecule_name.lower()}_rotation.mp4")
        
        if os.path.exists(video_path):
            media_url = QUrl.fromLocalFile(os.path.abspath(video_path))
            media_content = QMediaContent(media_url)
            self.media_player.setMedia(media_content)
            
            # Loop the video
            self.media_player.setNotifyInterval(100)
            self.media_player.play()
            
            print(f"[VideoMol] Loading video: {video_path}")
        else:
            print(f"[VideoMol] Video not found: {video_path}")
            self.show_placeholder(molecule_name)
    
    def show_placeholder(self, molecule_name):
        """Show a placeholder when video is not available."""
        # For now, we'll create placeholder videos or show static content
        print(f"[VideoMol] Showing placeholder for {molecule_name}")
    
    def on_media_status_changed(self, status):
        """Handle media status changes."""
        if status == QMediaPlayer.EndOfMedia:
            # Loop the video
            self.media_player.setPosition(0)
            self.media_player.play()
    
    def on_media_error(self, error):
        """Handle media player errors."""
        print(f"[VideoMol] Media error: {error}")

class AtmosphericVideoWindow(QMainWindow):
    """Main window with video-based molecular visualization."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Molecular Visualizer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Frameless window with custom styling
        self.setWindowFlags(Qt.FramelessWindowHint)
        
        # Set up central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Dark atmospheric styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: rgb(12, 8, 20);
                border: 3px solid rgb(120, 60, 180);
                border-radius: 20px;
            }
            QWidget {
                background-color: rgb(12, 8, 20);
            }
        """)
        
        self.setup_ui()
        self.center_window()
        
        print("[VideoMol] Atmospheric window created")
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self.central_widget)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Title
        title_label = QLabel("Molecular Video Visualizer")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: rgb(200, 150, 255);
                font-size: 24px;
                font-weight: bold;
                padding: 10px;
                background-color: rgba(30, 20, 50, 100);
                border-radius: 10px;
                border: 1px solid rgb(100, 50, 150);
            }
        """)
        layout.addWidget(title_label)
        
        # Dropdown section
        dropdown_layout = QHBoxLayout()
        dropdown_layout.addStretch()
        
        dropdown_label = QLabel("Select Molecule:")
        dropdown_label.setStyleSheet("""
            QLabel {
                color: rgb(180, 140, 230);
                font-size: 16px;
                font-weight: bold;
                padding: 5px;
            }
        """)
        dropdown_layout.addWidget(dropdown_label)
        
        self.molecule_dropdown = QComboBox()
        self.molecule_dropdown.addItems(["H2", "NH3", "CH4", "H2O", "CO2", "C2H6"])
        self.molecule_dropdown.setCurrentText("H2")
        
        # Enhanced dropdown styling
        self.molecule_dropdown.setStyleSheet("""
            QComboBox {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 rgb(40, 30, 70),
                                          stop: 1 rgb(25, 15, 45));
                color: white;
                border: 2px solid rgb(120, 80, 200);
                border-radius: 12px;
                padding: 12px 20px;
                font-size: 16px;
                font-weight: bold;
                min-width: 150px;
            }
            QComboBox:hover {
                border-color: rgb(150, 100, 250);
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 rgb(50, 40, 80),
                                          stop: 1 rgb(35, 25, 55));
            }
            QComboBox:pressed {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 rgb(30, 20, 60),
                                          stop: 1 rgb(20, 10, 40));
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 8px solid transparent;
                border-right: 8px solid transparent;
                border-top: 8px solid white;
                margin-right: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: rgb(30, 20, 50);
                color: white;
                border: 2px solid rgb(120, 80, 200);
                border-radius: 8px;
                selection-background-color: rgb(80, 40, 120);
            }
        """)
        
        dropdown_layout.addWidget(self.molecule_dropdown)
        dropdown_layout.addStretch()
        layout.addLayout(dropdown_layout)
        
        # Video viewer
        self.video_viewer = VideoMolecularViewer()
        layout.addWidget(self.video_viewer, 1)
        
        # Status bar
        self.status_label = QLabel("Ready - Select a molecule to view")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                color: rgb(150, 120, 200);
                font-size: 14px;
                padding: 8px;
                background-color: rgba(20, 15, 35, 150);
                border-radius: 8px;
                border: 1px solid rgb(80, 40, 120);
            }
        """)
        layout.addWidget(self.status_label)
        
        # Connect signals
        self.molecule_dropdown.currentTextChanged.connect(self.on_molecule_changed)
        
        # Load initial molecule
        self.on_molecule_changed("H2")
        
        print("[VideoMol] UI setup complete")
    
    def on_molecule_changed(self, molecule_name):
        """Handle molecule selection change."""
        self.video_viewer.load_molecule_video(molecule_name)
        self.status_label.setText(f"Displaying: {molecule_name}")
        print(f"[VideoMol] Switched to molecule: {molecule_name}")
    
    def center_window(self):
        """Center the window on screen."""
        screen = QApplication.desktop().screenGeometry()
        window = self.geometry()
        x = (screen.width() - window.width()) // 2
        y = (screen.height() - window.height()) // 2
        self.move(x, y)
    
    def mousePressEvent(self, event):
        """Enable window dragging."""
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        """Handle window dragging."""
        if event.buttons() == Qt.LeftButton and hasattr(self, 'drag_position'):
            self.move(event.globalPos() - self.drag_position)
            event.accept()
    
    def keyPressEvent(self, event):
        """Handle key presses."""
        if event.key() == Qt.Key_Escape:
            self.close()

def main():
    """Main application entry point."""
    try:
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        
        # Create main window
        window = AtmosphericVideoWindow()
        window.show()
        
        print("[VideoMol] Application started successfully")
        print("[VideoMol] Note: Create videos in gui/videos/ directory")
        print("[VideoMol] Expected format: {molecule}_rotation.mp4")
        
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"[VideoMol] Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
