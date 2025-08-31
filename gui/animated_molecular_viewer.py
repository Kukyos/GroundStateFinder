#!/usr/bin/env python3
"""Clean animated molecular viewer (restored)
Baseline stable version with:
- Multiple molecules (H2, NH3, H2O)
- Phases: welcome -> fade_transition -> running -> molecule_transition
- Crossfade transitions & brief flash
- Subtle glow & performance toggle
"""
import sys, math, random, time
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QComboBox
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont, QRadialGradient, QPainterPath


class AnimatedMolecularViewer(QWidget):
    def __init__(self):
        super().__init__()
        # Phase control
        self.phase = 'welcome'
        self.welcome_opacity = 1.0
        self.main_opacity = 0.0
        self.transition_timer = 0
        self.welcome_duration = 60
        self.fade_duration = 60
        # Molecule transition
        self.old_molecules = []
        self.transition_opacity = 1.0
        self.transition_duration = 12
        self.transition_flash = 0.0
        # Animation state
        self.current_molecule = 'H2'
        self.rotation_angle = 0
        self.ring_rotation_1 = 0
        self.ring_rotation_2 = 0
        self.ring_rotation_3 = 0
        self.pulse_angle = 0
        self.animation_time = 0
        self.orbital_time = 0
        self.orbital_speed = 0.008
        # Performance adapt
        self.performance_mode = False
        self._slow_frames = 0
        self._fast_frames = 0
        # Molecules
        self.molecules = []
        random.seed(int(time.time()*1000) % 100000)
        self.generate_random_molecules()
        # Visual config
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)
        self.setMinimumSize(1200, 900)
        # Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(33)

    # -------- Molecule Generation --------
    def generate_random_molecules(self):
        self.molecules.clear()
        types = ['H2', 'NH3', 'H2O']
        max_x, max_y = 600, 450
        zones = [
            (-max_x, -200, -max_y, -150), (200, max_x, -max_y, -150), (-150, 150, -max_y, -200),
            (-max_x, -250, -100, 100), (250, max_x, -100, 100),
            (-max_x, -200, 150, max_y), (200, max_x, 150, max_y), (-150, 150, 200, max_y),
            (-400, -150, -300, -100), (150, 400, -300, -100), (-400, -150, 100, 300), (150, 400, 100, 300)
        ]
        random.shuffle(zones)
        count = random.randint(4, 6)
        for i in range(count):
            zx1, zx2, zy1, zy2 = zones[i % len(zones)]
            x = random.randint(zx1, zx2)
            y = random.randint(zy1, zy2)
            x = max(-max_x+50, min(max_x-50, x + random.randint(-15,15)))
            y = max(-max_y+50, min(max_y-50, y + random.randint(-15,15)))
            self.molecules.append({
                'type': random.choice(types),
                'x_offset': x,
                'y_offset': y,
                'scale': random.uniform(0.75, 1.0),
                'rotation_offset': random.randint(0, 360),
                'speed_multiplier': random.uniform(0.85, 1.15)
            })

    # -------- Public API --------
    def set_molecule(self, name):
        if name == self.current_molecule:
            return
        self.current_molecule = name
        self.old_molecules = self.molecules.copy()
        self.generate_random_molecules()
        self.phase = 'molecule_transition'
        self.transition_timer = 0
        self.transition_opacity = 0.0
        self.transition_flash = 1.0
        self.update()

    # -------- Animation Update --------
    def update_animation(self):
        self.animation_time += 1
        self.orbital_time += self.orbital_speed
        t = self.animation_time * 0.005
        self.rotation_angle = 180 + 180 * math.sin(t)
        self.ring_rotation_1 = (self.animation_time * 0.30) % 360
        self.ring_rotation_2 = (self.animation_time * -0.22) % 360
        self.ring_rotation_3 = (self.animation_time * 0.41) % 360
        self.pulse_angle = (self.animation_time * 0.8) % 360
        # Phase logic
        if self.phase == 'welcome':
            self.transition_timer += 1
            if self.transition_timer >= self.welcome_duration:
                self.phase = 'fade_transition'; self.transition_timer = 0
        elif self.phase == 'fade_transition':
            self.transition_timer += 1
            p = min(1.0, self.transition_timer / self.fade_duration)
            ease = 1 - (1-p)**3
            self.welcome_opacity = 1 - ease
            self.main_opacity = ease
            if p >= 1.0:
                self.phase = 'running'; self.main_opacity = 1.0; self.welcome_opacity = 0.0
        elif self.phase == 'molecule_transition':
            self.transition_timer += 1
            p = min(1.0, self.transition_timer / self.transition_duration)
            self.transition_opacity = p*p*(3-2*p)
            if p >= 1.0:
                self.phase = 'running'; self.old_molecules = []; self.transition_opacity = 1.0
        self.update()

    # -------- Painting --------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        start_t = time.perf_counter()
        # Molecules / transitions
        if self.phase == 'molecule_transition':
            if self.old_molecules:
                painter.setOpacity(1.0 - self.transition_opacity)
                self.draw_molecules(painter, self.old_molecules)
            painter.setOpacity(self.transition_opacity)
            self.draw_molecules(painter, self.molecules)
            painter.setOpacity(1.0)
        else:
            painter.setOpacity(self.main_opacity if self.phase in ('fade_transition','running') else 0.0)
            self.draw_molecules(painter, self.molecules)
            painter.setOpacity(1.0)
        # Welcome overlay
        if self.phase in ('welcome','fade_transition'):
            self.draw_welcome(painter)
        # Flash
        if self.transition_flash > 0.01:
            a = int(85 * self.transition_flash)
            g = QRadialGradient(self.width()/2, self.height()/2, max(self.width(), self.height())*0.55)
            g.setColorAt(0.0, QColor(200,140,255,a))
            g.setColorAt(0.4, QColor(140,70,220,int(a*0.55)))
            g.setColorAt(1.0, QColor(20,10,40,0))
            painter.setOpacity(0.55)
            painter.fillRect(self.rect(), QBrush(g))
            painter.setOpacity(1.0)
            self.transition_flash = max(0.0, self.transition_flash - 0.12)
        # Perf adapt
        dt = time.perf_counter() - start_t
        if dt > 0.04:
            self._slow_frames += 1; self._fast_frames = 0
            if self._slow_frames > 6 and not self.performance_mode:
                self.performance_mode = True
        else:
            self._fast_frames += 1; self._slow_frames = 0
            if self._fast_frames > 40 and self.performance_mode:
                self.performance_mode = False
        painter.end()

    # -------- Molecule Drawing --------
    def draw_molecules(self, painter, mol_list):
        if not mol_list: return
        cx, cy = self.width()//2, self.height()//2
        painter.save()
        painter.translate(cx, cy)
        for m in mol_list:
            painter.save()
            painter.translate(m['x_offset'], m['y_offset'])
            painter.scale(m['scale'], m['scale'])
            painter.rotate(m['rotation_offset'] + self.rotation_angle * 0.25 * m['speed_multiplier'])
            t = m['type']
            if t == 'H2':
                self.draw_h2(painter)
            elif t == 'NH3':
                self.draw_nh3(painter)
            elif t == 'H2O':
                self.draw_h2o(painter)
            painter.restore()
        painter.restore()

    def nucleus_glow(self, painter, r, color):
        layers = 3 if self.performance_mode else 6
        for i in range(layers):
            g = QColor(color); g.setAlpha(90 - i*12)
            size = r*2 + i*10
            painter.setBrush(QBrush(g)); painter.setPen(Qt.NoPen)
            painter.drawEllipse(-size//2, -size//2, size, size)
        painter.setBrush(QBrush(color)); painter.setPen(QPen(color.darker(140),1))
        painter.drawEllipse(-r, -r, r*2, r*2)

    def draw_e_rings(self, painter, base_r, electrons, tilt_deg, rot):
        tilt = math.radians(tilt_deg)
        rw = base_r
        rh = max(4, int(base_r * abs(math.cos(tilt))))
        depth_iter = range(-1,2) if self.performance_mode else range(-2,3)
        for d in depth_iter:
            w = rw + d*1.2; h = rh + d*0.3
            alpha = max(35 - abs(d)*8, 12)
            pen = QPen(QColor(200,200,220,alpha), 1)
            painter.setPen(pen); painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(int(-w), int(-h), int(w*2), int(h*2))
        for i in range(electrons):
            angle = math.radians(rot + i * 360 / electrons)
            ex = rw * math.cos(angle)
            ey = rh * math.sin(angle)
            painter.setBrush(QBrush(QColor(255,255,255,230)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(int(ex-4), int(ey-4), 8, 8)
            if not self.performance_mode:
                painter.setBrush(QBrush(QColor(255,255,255,40)))
                painter.drawEllipse(int(ex-10), int(ey-10), 20, 20)

    def draw_h2(self, painter):
        painter.save(); painter.translate(-60,0); self.nucleus_glow(painter, 18, QColor(200,220,255)); painter.restore()
        painter.save(); painter.translate(60,0); self.nucleus_glow(painter, 18, QColor(200,220,255)); painter.restore()
        painter.save(); self.draw_e_rings(painter, 110, 4, 25, self.ring_rotation_1); painter.restore()

    def draw_nh3(self, painter):
        self.nucleus_glow(painter, 30, QColor(180,200,255))
        for i in range(3):
            ang = math.radians(i*120 + self.rotation_angle*0.2)
            painter.save(); painter.translate(150*math.cos(ang), 110*math.sin(ang))
            self.nucleus_glow(painter, 18, QColor(200,220,255))
            painter.restore()
        self.draw_e_rings(painter, 170, 6, 40, self.ring_rotation_2)
        self.draw_e_rings(painter, 260, 2, -35, self.ring_rotation_3)

    def draw_h2o(self, painter):
        self.nucleus_glow(painter, 34, QColor(170,200,255))
        for ang in (-55, 55):
            a = math.radians(ang + self.rotation_angle*0.15)
            painter.save(); painter.translate(190*math.cos(a), 140*math.sin(a))
            self.nucleus_glow(painter, 20, QColor(200,220,255))
            painter.restore()
        self.draw_e_rings(painter, 190, 6, 30, self.ring_rotation_1)
        self.draw_e_rings(painter, 285, 2, -40, self.ring_rotation_2)

    def draw_welcome(self, painter):
        painter.save(); painter.setOpacity(self.welcome_opacity)
        g = QRadialGradient(self.width()/2, self.height()/2, min(self.width(), self.height())/2)
        g.setColorAt(0.0, QColor(70,0,110,90))
        g.setColorAt(0.5, QColor(130,40,200,50))
        g.setColorAt(1.0, QColor(0,0,0,0))
        painter.fillRect(self.rect(), QBrush(g))
        painter.translate(self.width()/2, self.height()/2)
        txt = 'Welcome'
        font = QFont('Arial', 72, QFont.Bold)
        painter.setFont(font)
        for i in range(5):
            a = 90 - i*15
            if a <= 0: continue
            painter.setPen(QColor(255,255,255,a))
            painter.drawText(-painter.fontMetrics().width(txt)//2, painter.fontMetrics().ascent()//2, txt)
        painter.restore()


class AnimatedMolecularWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMinimumSize(1400, 1000)
        self.viewer = AnimatedMolecularViewer()
        self.setCentralWidget(self.viewer)
        # Dropdown overlay
        self.dropdown = QComboBox(self)
        self.dropdown.addItems(['H2','NH3','H2O'])
        self.dropdown.move(40,40)
        self.dropdown.resize(160,48)
        self.dropdown.setStyleSheet("""
            QComboBox {background:rgba(30,15,55,210);color:rgb(230,210,255);border:2px solid rgb(150,80,220);border-radius:14px;padding:10px 14px;font-size:16px;font-weight:bold;}
            QComboBox:hover {border-color:rgb(200,120,255);background:rgba(45,25,80,220);}
            QComboBox::drop-down {width:28px;border:0;}
            QComboBox::down-arrow {image:none;border-left:7px solid transparent; border-right:7px solid transparent; border-top:9px solid white; margin-right:6px;}
            QComboBox QAbstractItemView {background:rgba(25,15,45,240);color:white;border:2px solid rgb(150,80,220);selection-background-color:rgba(160,100,230,160);border-radius:10px;padding:6px;}
        """)
        self.dropdown.currentTextChanged.connect(self.viewer.set_molecule)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        r = self.rect()
        radius = 28
        rectf = QRectF(r.adjusted(2,2,-2,-2))
        path = QPainterPath(); path.addRoundedRect(rectf, radius, radius)
        glow = QRadialGradient(r.center(), r.width()*0.7)
        glow.setColorAt(0.0, QColor(120,60,200,60))
        glow.setColorAt(1.0, QColor(20,5,40,0))
        painter.fillRect(r, QBrush(glow))
        painter.setClipPath(path)
        painter.fillRect(r, QColor(10,5,20,220))
        painter.setClipping(False)
        painter.setPen(QPen(QColor(160,100,240,180), 3))
        painter.drawPath(path)
        painter.end()


def main():
    app = QApplication(sys.argv)
    win = AnimatedMolecularWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
