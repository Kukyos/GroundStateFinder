#!/usr/bin/env python3
"""
Modern 3D Molecular Visualizer using QOpenGLWidget
This uses the modern Qt OpenGL approach instead of the deprecated QGLWidget
"""
import sys
import math
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QComboBox, QHBoxLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QRegion, QPalette
from PyQt5.QtOpenGL import QOpenGLWidget
import OpenGL.GL as gl

print("[Modern3D] Starting modern 3D molecular visualizer...")

class ModernMolecularViewer(QOpenGLWidget):
    """Modern OpenGL molecular viewer using QOpenGLWidget."""
    
    def __init__(self):
        super().__init__()
        self.rotation_x = 0
        self.rotation_y = 0
        self.current_molecule = "H2"
        
        # Set widget properties for solid background
        self.setAutoFillBackground(True)
        
        # Set up animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.rotate_molecule)
        self.timer.start(50)  # 20 FPS
        
        print("[Modern3D] Widget created with animation timer")
    
    def initializeGL(self):
        """Initialize OpenGL context with proper error handling."""
        try:
            print("[Modern3D] Initializing OpenGL context...")
            
            # Set clear color to dark blue-black
            gl.glClearColor(0.02, 0.01, 0.08, 1.0)
            
            # Enable depth testing
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthFunc(gl.GL_LESS)
            
            # Enable basic lighting
            gl.glEnable(gl.GL_LIGHTING)
            gl.glEnable(gl.GL_LIGHT0)
            
            # Set light position
            light_pos = [1.0, 1.0, 1.0, 0.0]
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, light_pos)
            
            # Set light colors
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, [0.2, 0.2, 0.3, 1.0])
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, [0.8, 0.8, 1.0, 1.0])
            
            # Enable color material
            gl.glEnable(gl.GL_COLOR_MATERIAL)
            gl.glColorMaterial(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE)
            
            print("[Modern3D] OpenGL context initialized successfully")
            
        except Exception as e:
            print(f"[Modern3D] OpenGL initialization error: {e}")
    
    def resizeGL(self, width, height):
        """Handle widget resize."""
        try:
            gl.glViewport(0, 0, width, height)
            
            # Set up perspective projection
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            
            # Simple perspective
            if height > 0:
                aspect = width / height
                fov = 45.0
                near = 0.1
                far = 100.0
                
                f = 1.0 / math.tan(math.radians(fov) / 2.0)
                gl.glFrustum(-aspect * near / f, aspect * near / f, 
                           -near / f, near / f, near, far)
            
            gl.glMatrixMode(gl.GL_MODELVIEW)
            print(f"[Modern3D] Viewport resized to {width}x{height}")
            
        except Exception as e:
            print(f"[Modern3D] Resize error: {e}")
    
    def paintGL(self):
        """Render the 3D molecular scene."""
        try:
            # Clear buffers
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            
            # Reset modelview matrix
            gl.glLoadIdentity()
            
            # Move back from camera
            gl.glTranslatef(0.0, 0.0, -3.0)
            
            # Apply rotations
            gl.glRotatef(self.rotation_x, 1.0, 0.0, 0.0)
            gl.glRotatef(self.rotation_y, 0.0, 1.0, 0.0)
            
            # Draw the selected molecule
            if self.current_molecule == "H2":
                self.draw_h2()
            elif self.current_molecule == "NH3":
                self.draw_nh3()
            elif self.current_molecule == "CH4":
                self.draw_ch4()
            else:
                self.draw_h2()  # Default
            
            # Force buffer swap
            gl.glFlush()
            
        except Exception as e:
            print(f"[Modern3D] Paint error: {e}")
    
    def draw_h2(self):
        """Draw H2 molecule (2 hydrogen atoms)."""
        try:
            # Left hydrogen (white)
            gl.glColor3f(0.9, 0.9, 1.0)
            gl.glPushMatrix()
            gl.glTranslatef(-0.5, 0.0, 0.0)
            self.draw_sphere(0.3)
            gl.glPopMatrix()
            
            # Right hydrogen (white)
            gl.glColor3f(0.9, 0.9, 1.0)
            gl.glPushMatrix()
            gl.glTranslatef(0.5, 0.0, 0.0)
            self.draw_sphere(0.3)
            gl.glPopMatrix()
            
            # Bond (yellow line)
            gl.glDisable(gl.GL_LIGHTING)
            gl.glColor3f(1.0, 1.0, 0.0)
            gl.glLineWidth(3.0)
            gl.glBegin(gl.GL_LINES)
            gl.glVertex3f(-0.5, 0.0, 0.0)
            gl.glVertex3f(0.5, 0.0, 0.0)
            gl.glEnd()
            gl.glEnable(gl.GL_LIGHTING)
            
        except Exception as e:
            print(f"[Modern3D] H2 drawing error: {e}")
    
    def draw_nh3(self):
        """Draw NH3 molecule (1 nitrogen + 3 hydrogens)."""
        try:
            # Nitrogen (blue)
            gl.glColor3f(0.2, 0.2, 1.0)
            gl.glPushMatrix()
            gl.glTranslatef(0.0, 0.0, 0.0)
            self.draw_sphere(0.4)
            gl.glPopMatrix()
            
            # Three hydrogens in pyramid
            positions = [
                [0.8, -0.5, 0.0],
                [-0.4, -0.5, 0.7],
                [-0.4, -0.5, -0.7]
            ]
            
            gl.glColor3f(0.9, 0.9, 1.0)
            for pos in positions:
                gl.glPushMatrix()
                gl.glTranslatef(pos[0], pos[1], pos[2])
                self.draw_sphere(0.3)
                gl.glPopMatrix()
            
            # Bonds
            gl.glDisable(gl.GL_LIGHTING)
            gl.glColor3f(1.0, 1.0, 0.0)
            gl.glLineWidth(3.0)
            gl.glBegin(gl.GL_LINES)
            for pos in positions:
                gl.glVertex3f(0.0, 0.0, 0.0)
                gl.glVertex3f(pos[0], pos[1], pos[2])
            gl.glEnd()
            gl.glEnable(gl.GL_LIGHTING)
            
        except Exception as e:
            print(f"[Modern3D] NH3 drawing error: {e}")
    
    def draw_ch4(self):
        """Draw CH4 molecule (1 carbon + 4 hydrogens)."""
        try:
            # Carbon (black/gray)
            gl.glColor3f(0.3, 0.3, 0.3)
            gl.glPushMatrix()
            gl.glTranslatef(0.0, 0.0, 0.0)
            self.draw_sphere(0.4)
            gl.glPopMatrix()
            
            # Four hydrogens in tetrahedral
            positions = [
                [0.7, 0.7, 0.7],
                [-0.7, -0.7, 0.7],
                [-0.7, 0.7, -0.7],
                [0.7, -0.7, -0.7]
            ]
            
            gl.glColor3f(0.9, 0.9, 1.0)
            for pos in positions:
                gl.glPushMatrix()
                gl.glTranslatef(pos[0], pos[1], pos[2])
                self.draw_sphere(0.3)
                gl.glPopMatrix()
            
            # Bonds
            gl.glDisable(gl.GL_LIGHTING)
            gl.glColor3f(1.0, 1.0, 0.0)
            gl.glLineWidth(3.0)
            gl.glBegin(gl.GL_LINES)
            for pos in positions:
                gl.glVertex3f(0.0, 0.0, 0.0)
                gl.glVertex3f(pos[0], pos[1], pos[2])
            gl.glEnd()
            gl.glEnable(gl.GL_LIGHTING)
            
        except Exception as e:
            print(f"[Modern3D] CH4 drawing error: {e}")
    
    def draw_sphere(self, radius):
        """Draw a simple sphere using triangle strips."""
        try:
            # Simple sphere approximation with quads
            slices = 16
            stacks = 12
            
            for i in range(stacks):
                lat1 = math.pi * (-0.5 + float(i) / stacks)
                lat2 = math.pi * (-0.5 + float(i + 1) / stacks)
                
                gl.glBegin(gl.GL_QUAD_STRIP)
                for j in range(slices + 1):
                    lng = 2 * math.pi * float(j) / slices
                    
                    x1 = radius * math.cos(lat1) * math.cos(lng)
                    y1 = radius * math.sin(lat1)
                    z1 = radius * math.cos(lat1) * math.sin(lng)
                    
                    x2 = radius * math.cos(lat2) * math.cos(lng)
                    y2 = radius * math.sin(lat2)
                    z2 = radius * math.cos(lat2) * math.sin(lng)
                    
                    gl.glNormal3f(x1/radius, y1/radius, z1/radius)
                    gl.glVertex3f(x1, y1, z1)
                    
                    gl.glNormal3f(x2/radius, y2/radius, z2/radius)
                    gl.glVertex3f(x2, y2, z2)
                
                gl.glEnd()
                
        except Exception as e:
            print(f"[Modern3D] Sphere drawing error: {e}")
    
    def rotate_molecule(self):
        """Continuous rotation animation."""
        self.rotation_x += 1.0
        self.rotation_y += 1.5
        if self.rotation_x >= 360:
            self.rotation_x -= 360
        if self.rotation_y >= 360:
            self.rotation_y -= 360
        self.update()  # Trigger repaint
    
    def set_molecule(self, molecule_name):
        """Change the displayed molecule."""
        self.current_molecule = molecule_name
        print(f"[Modern3D] Switched to molecule: {molecule_name}")
        self.update()

class AtmosphericWindow(QMainWindow):
    """Main window with atmospheric styling."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modern 3D Molecular Visualizer")
        self.setGeometry(100, 100, 1000, 700)
        
        # Set window flags for modern appearance
        self.setWindowFlags(Qt.FramelessWindowHint)
        
        # Set up central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Set dark background
        self.setStyleSheet("""
            QMainWindow {
                background-color: rgb(12, 8, 20);
                border: 2px solid rgb(100, 50, 150);
                border-radius: 15px;
            }
        """)
        
        self.setup_ui()
        
        # Center the window
        self.center_window()
        
        print("[Modern3D] Window created and centered")
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self.central_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Create dropdown for molecule selection
        dropdown_layout = QHBoxLayout()
        dropdown_layout.addStretch()
        
        self.molecule_dropdown = QComboBox()
        self.molecule_dropdown.addItems(["H2", "NH3", "CH4"])
        self.molecule_dropdown.setCurrentText("H2")
        
        # Style the dropdown
        self.molecule_dropdown.setStyleSheet("""
            QComboBox {
                background-color: rgba(30, 20, 50, 180);
                color: white;
                border: 2px solid rgb(120, 80, 200);
                border-radius: 8px;
                padding: 8px 15px;
                font-size: 14px;
                font-weight: bold;
                min-width: 120px;
            }
            QComboBox:hover {
                border-color: rgb(150, 100, 250);
                background-color: rgba(40, 30, 70, 200);
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid white;
                margin-right: 5px;
            }
        """)
        
        dropdown_layout.addWidget(self.molecule_dropdown)
        dropdown_layout.addStretch()
        layout.addLayout(dropdown_layout)
        
        # Create the 3D viewer
        self.viewer_3d = ModernMolecularViewer()
        layout.addWidget(self.viewer_3d, 1)  # Give it most of the space
        
        # Connect dropdown to viewer
        self.molecule_dropdown.currentTextChanged.connect(self.viewer_3d.set_molecule)
        
        print("[Modern3D] UI setup complete")
    
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

def main():
    """Main application entry point."""
    try:
        app = QApplication(sys.argv)
        
        # Set application style
        app.setStyle('Fusion')
        
        # Create and show main window
        window = AtmosphericWindow()
        window.show()
        
        print("[Modern3D] Application started successfully")
        
        # Run the application
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"[Modern3D] Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
