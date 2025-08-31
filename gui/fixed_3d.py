#!/usr/bin/env python3
"""
Fixed 3D Molecular Visualizer with proper OpenGL context handling
This version addresses the "No GL context" issue by ensuring proper initialization
"""
import sys
import math
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QComboBox, QHBoxLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QRegion, QPalette
from PyQt5.QtOpenGL import QGLWidget, QGLFormat

# Try to import OpenGL - with better error handling
try:
    import OpenGL.GL as gl
    import OpenGL.GLU as glu
    OPENGL_AVAILABLE = True
    print("[Fixed3D] OpenGL imported successfully")
except ImportError as e:
    print(f"[Fixed3D] OpenGL import failed: {e}")
    OPENGL_AVAILABLE = False

print("[Fixed3D] Starting fixed 3D molecular visualizer...")

class FixedMolecularViewer(QGLWidget):
    """Fixed OpenGL molecular viewer with proper context handling."""
    
    def __init__(self):
        # Create a specific OpenGL format for better compatibility
        fmt = QGLFormat()
        fmt.setDoubleBuffer(True)
        fmt.setDepth(True)
        fmt.setRgba(True)
        fmt.setSampleBuffers(True)
        fmt.setVersion(2, 1)  # Request OpenGL 2.1
        
        super().__init__(fmt)
        
        self.rotation_x = 0
        self.rotation_y = 0
        self.current_molecule = "H2"
        self.gl_initialized = False
        
        # Set widget properties
        self.setAutoFillBackground(True)
        self.setMinimumSize(400, 400)
        
        # Set dark background via stylesheet
        self.setStyleSheet("""
            QGLWidget {
                background-color: rgb(12, 8, 20);
                border: 1px solid rgb(80, 40, 120);
            }
        """)
        
        # Set up animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.rotate_molecule)
        
        print("[Fixed3D] Widget created with specific OpenGL format")
    
    def initializeGL(self):
        """Initialize OpenGL context with extensive error checking."""
        print("[Fixed3D] initializeGL called...")
        
        if not OPENGL_AVAILABLE:
            print("[Fixed3D] OpenGL not available - using fallback rendering")
            self.gl_initialized = False
            return
        
        try:
            # Check if we have a valid context
            if not self.isValid():
                print("[Fixed3D] Invalid OpenGL context!")
                return
            
            # Make context current
            self.makeCurrent()
            
            # Get OpenGL info
            vendor = gl.glGetString(gl.GL_VENDOR)
            renderer = gl.glGetString(gl.GL_RENDERER)
            version = gl.glGetString(gl.GL_VERSION)
            print(f"[Fixed3D] OpenGL Vendor: {vendor}")
            print(f"[Fixed3D] OpenGL Renderer: {renderer}")
            print(f"[Fixed3D] OpenGL Version: {version}")
            
            # Set clear color to dark blue-black
            gl.glClearColor(0.02, 0.01, 0.08, 1.0)
            
            # Enable depth testing
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthFunc(gl.GL_LESS)
            
            # Try to enable lighting with error checking
            try:
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
                
                print("[Fixed3D] Lighting enabled successfully")
            except Exception as lighting_error:
                print(f"[Fixed3D] Lighting setup failed: {lighting_error}")
            
            self.gl_initialized = True
            print("[Fixed3D] OpenGL context initialized successfully")
            
            # Start animation after successful initialization
            if not self.timer.isActive():
                self.timer.start(50)  # 20 FPS
                print("[Fixed3D] Animation timer started")
            
        except Exception as e:
            print(f"[Fixed3D] OpenGL initialization error: {e}")
            self.gl_initialized = False
    
    def resizeGL(self, width, height):
        """Handle widget resize."""
        if not self.gl_initialized or not OPENGL_AVAILABLE:
            return
        
        try:
            self.makeCurrent()
            gl.glViewport(0, 0, width, height)
            
            # Set up perspective projection
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            
            if height > 0:
                glu.gluPerspective(45.0, width / height, 0.1, 100.0)
            
            gl.glMatrixMode(gl.GL_MODELVIEW)
            print(f"[Fixed3D] Viewport resized to {width}x{height}")
            
        except Exception as e:
            print(f"[Fixed3D] Resize error: {e}")
    
    def paintGL(self):
        """Render the 3D molecular scene."""
        if not self.gl_initialized or not OPENGL_AVAILABLE:
            # Fallback to QPainter rendering
            self.paintFallback()
            return
        
        try:
            self.makeCurrent()
            
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
            self.swapBuffers()
            
        except Exception as e:
            print(f"[Fixed3D] Paint error: {e}")
    
    def paintFallback(self):
        """Fallback 2D rendering when OpenGL is not available."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Fill background
        painter.fillRect(self.rect(), QColor(12, 8, 20))
        
        # Draw a simple molecular representation
        center_x = self.width() // 2
        center_y = self.height() // 2
        
        # Rotate the 2D representation
        painter.translate(center_x, center_y)
        painter.rotate(self.rotation_y * 0.5)
        
        if self.current_molecule == "H2":
            # Two atoms connected by a bond
            painter.setBrush(QBrush(QColor(200, 200, 255)))  # Light blue for hydrogen
            painter.setPen(QPen(QColor(255, 255, 0), 3))  # Yellow bond
            
            # Draw bond
            painter.drawLine(-50, 0, 50, 0)
            
            # Draw atoms
            painter.drawEllipse(-60, -15, 30, 30)
            painter.drawEllipse(30, -15, 30, 30)
            
        elif self.current_molecule == "NH3":
            # Central nitrogen with three hydrogens
            painter.setBrush(QBrush(QColor(50, 50, 255)))  # Blue nitrogen
            painter.drawEllipse(-20, -20, 40, 40)
            
            painter.setBrush(QBrush(QColor(200, 200, 255)))  # Light blue hydrogens
            painter.setPen(QPen(QColor(255, 255, 0), 2))
            
            # Draw bonds and hydrogens
            positions = [(60, -30), (-30, 50), (-30, -50)]
            for pos in positions:
                painter.drawLine(0, 0, pos[0], pos[1])
                painter.drawEllipse(pos[0]-10, pos[1]-10, 20, 20)
        
        elif self.current_molecule == "CH4":
            # Central carbon with four hydrogens
            painter.setBrush(QBrush(QColor(80, 80, 80)))  # Gray carbon
            painter.drawEllipse(-20, -20, 40, 40)
            
            painter.setBrush(QBrush(QColor(200, 200, 255)))  # Light blue hydrogens
            painter.setPen(QPen(QColor(255, 255, 0), 2))
            
            # Draw bonds and hydrogens in tetrahedral-like pattern
            positions = [(50, -50), (-50, -50), (-50, 50), (50, 50)]
            for pos in positions:
                painter.drawLine(0, 0, pos[0], pos[1])
                painter.drawEllipse(pos[0]-10, pos[1]-10, 20, 20)
        
        painter.end()
    
    def draw_h2(self):
        """Draw H2 molecule (2 hydrogen atoms)."""
        if not OPENGL_AVAILABLE:
            return
        
        try:
            # Left hydrogen (white)
            gl.glColor3f(0.9, 0.9, 1.0)
            gl.glPushMatrix()
            gl.glTranslatef(-0.5, 0.0, 0.0)
            glu.gluSphere(glu.gluNewQuadric(), 0.3, 16, 16)
            gl.glPopMatrix()
            
            # Right hydrogen (white)
            gl.glColor3f(0.9, 0.9, 1.0)
            gl.glPushMatrix()
            gl.glTranslatef(0.5, 0.0, 0.0)
            glu.gluSphere(glu.gluNewQuadric(), 0.3, 16, 16)
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
            print(f"[Fixed3D] H2 drawing error: {e}")
    
    def draw_nh3(self):
        """Draw NH3 molecule (1 nitrogen + 3 hydrogens)."""
        if not OPENGL_AVAILABLE:
            return
        
        try:
            # Nitrogen (blue)
            gl.glColor3f(0.2, 0.2, 1.0)
            gl.glPushMatrix()
            gl.glTranslatef(0.0, 0.0, 0.0)
            glu.gluSphere(glu.gluNewQuadric(), 0.4, 16, 16)
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
                glu.gluSphere(glu.gluNewQuadric(), 0.3, 16, 16)
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
            print(f"[Fixed3D] NH3 drawing error: {e}")
    
    def draw_ch4(self):
        """Draw CH4 molecule (1 carbon + 4 hydrogens)."""
        if not OPENGL_AVAILABLE:
            return
        
        try:
            # Carbon (black/gray)
            gl.glColor3f(0.3, 0.3, 0.3)
            gl.glPushMatrix()
            gl.glTranslatef(0.0, 0.0, 0.0)
            glu.gluSphere(glu.gluNewQuadric(), 0.4, 16, 16)
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
                glu.gluSphere(glu.gluNewQuadric(), 0.3, 16, 16)
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
            print(f"[Fixed3D] CH4 drawing error: {e}")
    
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
        print(f"[Fixed3D] Switched to molecule: {molecule_name}")
        self.update()

class FixedAtmosphericWindow(QMainWindow):
    """Main window with atmospheric styling and proper OpenGL handling."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fixed 3D Molecular Visualizer")
        self.setGeometry(100, 100, 1000, 700)
        
        # Set window flags but keep transparency control
        self.setWindowFlags(Qt.FramelessWindowHint)
        
        # Set up central widget with solid background
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Solid background styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: rgb(12, 8, 20);
                border: 2px solid rgb(100, 50, 150);
                border-radius: 15px;
            }
            QWidget {
                background-color: rgb(12, 8, 20);
            }
        """)
        
        self.setup_ui()
        self.center_window()
        
        print("[Fixed3D] Window created with solid background")
    
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
        
        # Style the dropdown with solid background
        self.molecule_dropdown.setStyleSheet("""
            QComboBox {
                background-color: rgb(30, 20, 50);
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
                background-color: rgb(40, 30, 70);
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
        self.viewer_3d = FixedMolecularViewer()
        layout.addWidget(self.viewer_3d, 1)  # Give it most of the space
        
        # Connect dropdown to viewer
        self.molecule_dropdown.currentTextChanged.connect(self.viewer_3d.set_molecule)
        
        print("[Fixed3D] UI setup complete")
    
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
        window = FixedAtmosphericWindow()
        window.show()
        
        print("[Fixed3D] Application started successfully")
        
        # Run the application
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"[Fixed3D] Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
