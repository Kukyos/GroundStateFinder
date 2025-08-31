#!/usr/bin/env python3
"""Atmospheric 3D Molecular Visualizer with Rounded Glowing Edges"""
import sys
import math
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import QPainter, QColor, QFont, QPainterPath, QRadialGradient, QRegion, QPen
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QComboBox, QVBoxLayout, QHBoxLayout
from PyQt5.QtOpenGL import QGLWidget

class Simple3DViewer(QGLWidget):
    """Simplified 3D molecular viewer that actually renders."""
    
    def __init__(self):
        super().__init__()
        self.molecule = None  # Start with no molecule selected
        self.angle = 0.0
        
        # Set solid background
        self.setAutoFillBackground(True)
        self.setStyleSheet("background-color: rgb(5, 3, 10);")  # Solid dark background
        
        # Animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_rotation)
        self.timer.start(50)  # 20 fps
        
        self.setMinimumSize(800, 800)
        print("[Simple3D] Widget created")

    def set_molecule(self, name):
        """Set molecule to render."""
        self.molecule = name
        print(f"[Simple3D] Molecule set to: {name}")
        self.updateGL()

    def update_rotation(self):
        """Update rotation angle."""
        self.angle += 2.0
        if self.angle >= 360:
            self.angle = 0
        self.updateGL()

    def initializeGL(self):
        """Set up OpenGL."""
        print("[Simple3D] initializeGL called")
        try:
            from OpenGL import GL
            self.GL = GL
            
            # Solid dark background - no transparency
            GL.glClearColor(0.02, 0.01, 0.05, 1.0)  # Solid dark space background
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDepthFunc(GL.GL_LESS)
            
            print("[Simple3D] OpenGL setup complete")
            
        except ImportError:
            print("[Simple3D] OpenGL import failed")
            self.GL = None

    def resizeGL(self, w, h):
        """Handle resize."""
        if not self.GL:
            return
            
        print(f"[Simple3D] resizeGL: {w}x{h}")
        self.GL.glViewport(0, 0, w, h)
        self.GL.glMatrixMode(self.GL.GL_PROJECTION)
        self.GL.glLoadIdentity()
        
        # Simple perspective
        try:
            from OpenGL import GLU
            GLU.gluPerspective(60.0, w/h, 0.1, 100.0)
        except ImportError:
            # Manual perspective if GLU not available
            pass
            
        self.GL.glMatrixMode(self.GL.GL_MODELVIEW)

    def paintGL(self):
        """Draw the scene."""
        if not self.GL:
            print("[Simple3D] No GL context in paintGL")
            return
            
        GL = self.GL
        print(f"[Simple3D] paintGL called, molecule={self.molecule}, angle={self.angle}")
        
        # Clear everything
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glLoadIdentity()
        
        # Move camera back
        GL.glTranslatef(0.0, 0.0, -5.0)
        
        # Rotate based on timer
        GL.glRotatef(self.angle, 0, 1, 0)
        GL.glRotatef(self.angle * 0.7, 1, 0, 0)
        
        # Draw based on molecule
        if self.molecule == "H2":
            self.draw_h2()
        elif self.molecule == "NH3":
            self.draw_nh3()
        else:
            self.draw_test()

    def draw_test(self):
        """Draw empty space - no initial geometry."""
        GL = self.GL
        print("[Simple3D] No molecule selected - rendering empty space")
        
        # Just clear to dark background, no geometry
        pass

    def draw_h2(self):
        """Draw H2 molecule."""
        GL = self.GL
        print("[Simple3D] Drawing H2")
        
        GL.glDisable(GL.GL_LIGHTING)
        
        # Bond (white line)
        GL.glColor3f(1.0, 1.0, 1.0)
        GL.glLineWidth(6.0)
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3f(-0.8, 0.0, 0.0)
        GL.glVertex3f(0.8, 0.0, 0.0)
        GL.glEnd()
        
        # H atoms (cyan spheres approximated as points)
        GL.glColor3f(0.0, 1.0, 1.0)
        GL.glPointSize(25.0)
        GL.glBegin(GL.GL_POINTS)
        GL.glVertex3f(-0.8, 0.0, 0.0)
        GL.glVertex3f(0.8, 0.0, 0.0)
        GL.glEnd()
        
        # Try actual spheres if GLU available
        try:
            from OpenGL import GLU
            GL.glEnable(GL.GL_LIGHTING)
            GL.glEnable(GL.GL_LIGHT0)
            
            # Simple lighting
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, [1, 1, 1, 0])
            
            quad = GLU.gluNewQuadric()
            GLU.gluQuadricNormals(quad, GLU.GLU_SMOOTH)
            
            GL.glColor3f(0.7, 0.9, 1.0)
            
            GL.glPushMatrix()
            GL.glTranslatef(-0.8, 0.0, 0.0)
            GLU.gluSphere(quad, 0.25, 16, 16)
            GL.glPopMatrix()
            
            GL.glPushMatrix()
            GL.glTranslatef(0.8, 0.0, 0.0)
            GLU.gluSphere(quad, 0.25, 16, 16)
            GL.glPopMatrix()
            
            GLU.gluDeleteQuadric(quad)
        except ImportError:
            pass

    def draw_nh3(self):
        """Draw NH3 molecule.""" 
        GL = self.GL
        print("[Simple3D] Drawing NH3")
        
        GL.glDisable(GL.GL_LIGHTING)
        
        # Nitrogen center (blue)
        GL.glColor3f(0.0, 0.0, 1.0)
        GL.glPointSize(30.0)
        GL.glBegin(GL.GL_POINTS)
        GL.glVertex3f(0.0, 0.0, 0.0)
        GL.glEnd()
        
        # Three H atoms
        positions = [(1.0, 0.0, 0.5), (-0.5, 0.866, 0.5), (-0.5, -0.866, 0.5)]
        
        # Bonds
        GL.glColor3f(1.0, 1.0, 1.0)
        GL.glLineWidth(4.0)
        GL.glBegin(GL.GL_LINES)
        for pos in positions:
            GL.glVertex3f(0.0, 0.0, 0.0)
            GL.glVertex3f(pos[0], pos[1], pos[2])
        GL.glEnd()
        
        # H atoms
        GL.glColor3f(1.0, 1.0, 0.0)  # Yellow
        GL.glPointSize(20.0)
        GL.glBegin(GL.GL_POINTS)
        for pos in positions:
            GL.glVertex3f(pos[0], pos[1], pos[2])
        GL.glEnd()

    def paintEvent(self, event):
        """Override paint event to ensure solid background."""
        if self.GL:
            # Let OpenGL handle rendering
            self.paintGL()
        else:
            # Fallback painter with solid background
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(5, 3, 10))  # Solid dark background
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Arial", 16))
            painter.drawText(self.rect(), Qt.AlignCenter, 
                f"OpenGL not available\nSelected: {self.molecule}")
            painter.end()


class RoundedWindow(QMainWindow):
    """Atmospheric rounded window with glowing purple edges."""
    
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        
        # DON'T set translucent background - this makes everything transparent!
        # self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Larger centered window
        self.setGeometry(200, 100, 1000, 900)
        
        # Center on screen
        screen = QApplication.desktop().screenGeometry()
        window_rect = self.geometry()
        x = (screen.width() - window_rect.width()) // 2
        y = (screen.height() - window_rect.height()) // 2
        self.move(x, y)
        
        # Enable dragging
        self.dragging = False
        self.drag_position = None
        
        # Create central widget
        self.central_widget = AtmosphericWidget()
        self.setCentralWidget(self.central_widget)
        
        # Set window mask for rounded edges
        self.update_mask()

    def update_mask(self):
        """Create rounded window mask."""
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), 20, 20)
        mask = QRegion(path.toFillPolygon().toPolygon())
        self.setMask(mask)

    def resizeEvent(self, event):
        """Update mask on resize."""
        super().resizeEvent(event)
        self.update_mask()

    def mousePressEvent(self, event):
        """Enable window dragging."""
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        """Handle window dragging."""
        if self.dragging and self.drag_position:
            self.move(event.globalPos() - self.drag_position)

    def mouseReleaseEvent(self, event):
        """End dragging."""
        self.dragging = False

    def keyPressEvent(self, event):
        """Handle close on Escape."""
        if event.key() == Qt.Key_Escape:
            self.close()


class AtmosphericWidget(QWidget):
    """Central widget with glowing background and translucent dropdown."""
    
    def __init__(self):
        super().__init__()
        # Ensure solid background
        self.setAutoFillBackground(True)
        self.setStyleSheet("background-color: rgb(12, 8, 20);")  # Solid dark background
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the atmospheric UI."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 40, 30, 30)
        
        # Top spacer for dropdown positioning
        layout.addStretch(1)
        
        # Dropdown container (centered)
        dropdown_layout = QHBoxLayout()
        dropdown_layout.addStretch(1)
        
        # Create the glowing dropdown
        self.dropdown = GlowingDropdown()
        self.dropdown.addItems(["Select Molecule", "H2", "NH3"])
        self.dropdown.setFixedSize(250, 50)
        dropdown_layout.addWidget(self.dropdown)
        dropdown_layout.addStretch(1)
        
        layout.addLayout(dropdown_layout)
        layout.addSpacing(30)
        
        # 3D viewer
        self.viewer = Simple3DViewer()
        layout.addWidget(self.viewer)
        
        # Connect dropdown
        self.dropdown.currentTextChanged.connect(self.molecule_changed)

    def molecule_changed(self, molecule):
        """Handle molecule selection."""
        print(f"[AtmosphericWidget] Molecule changed to: {molecule}")
        if molecule != "Select Molecule":
            self.viewer.set_molecule(molecule)
        else:
            self.viewer.set_molecule(None)

    def paintEvent(self, event):
        """Paint the glowing rounded background."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Create the main rounded rectangle
        rect = QRectF(self.rect()).adjusted(10, 10, -10, -10)
        
        # Outer glow effect
        for i in range(15):
            alpha = 30 - i * 2
            if alpha <= 0:
                break
            
            glow_rect = rect.adjusted(-i*2, -i*2, i*2, i*2)
            painter.setPen(QPen(QColor(120, 80, 200, alpha), 2))
            painter.setBrush(Qt.NoBrush)
            
            path = QPainterPath()
            path.addRoundedRect(glow_rect, 20 + i, 20 + i)
            painter.drawPath(path)
        
        # Solid dark background (no transparency in the middle!)
        painter.setBrush(QColor(12, 8, 20))  # Solid dark background
        painter.setPen(QPen(QColor(150, 100, 220), 3))
        
        path = QPainterPath()
        path.addRoundedRect(rect, 18, 18)
        painter.drawPath(path)


class GlowingDropdown(QComboBox):
    """Dropdown with glowing edges and translucent styling."""
    
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QComboBox {
                background-color: rgba(40, 30, 60, 240);
                color: rgb(220, 200, 255);
                border: 2px solid rgba(140, 100, 200, 255);
                border-radius: 12px;
                padding: 12px 16px;
                font-size: 16px;
                font-weight: bold;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QComboBox:hover {
                background-color: rgba(60, 45, 90, 255);
                border: 2px solid rgba(160, 120, 220, 255);
            }
            QComboBox:focus {
                background-color: rgba(70, 55, 110, 255);
                border: 3px solid rgba(180, 140, 240, 255);
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 6px solid transparent;
                border-right: 6px solid transparent;
                border-top: 8px solid rgb(220, 200, 255);
                margin-right: 8px;
            }
            QComboBox QAbstractItemView {
                background-color: rgba(50, 35, 80, 250);
                color: rgb(220, 200, 255);
                selection-background-color: rgba(120, 90, 160, 255);
                border: 2px solid rgba(140, 100, 200, 255);
                border-radius: 8px;
                padding: 8px;
                font-size: 15px;
            }
            QComboBox QAbstractItemView::item {
                padding: 8px 12px;
                border-radius: 6px;
                margin: 2px;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: rgba(140, 110, 180, 255);
            }
        """)

    def paintEvent(self, event):
        """Add glow effect around dropdown."""
        # Draw glow effect first
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = QRectF(self.rect())
        
        # Glow effect
        for i in range(8):
            alpha = 40 - i * 5
            if alpha <= 0:
                break
                
            glow_rect = rect.adjusted(-i, -i, i, i)
            painter.setPen(QPen(QColor(160, 120, 220, alpha), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(glow_rect, 12 + i, 12 + i)
        
        # Call parent paint event for normal rendering
        super().paintEvent(event)


def main():
    print("[Main] Starting atmospheric 3D molecular visualizer...")
    app = QApplication(sys.argv)
    
    window = RoundedWindow()
    window.show()
    
    print("[Main] Window shown, starting event loop...")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
