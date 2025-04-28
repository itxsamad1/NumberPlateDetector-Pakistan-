import cv2
import numpy as np
import os
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QWidget, QListWidget, 
                            QListWidgetItem, QTextEdit, QFileDialog, QTabWidget,
                            QMessageBox, QSlider)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

class PlateDetector:
    def __init__(self):
        # Load the Pakistani license plate cascade classifier
        self.cascade_path = "pak.xml"
        self.plate_cascade = cv2.CascadeClassifier(self.cascade_path)
        if self.plate_cascade.empty():
            print(f"Error: Failed to load cascade classifier from {self.cascade_path}")
            raise ValueError("Cascade classifier not loaded")

    def detect_plates(self, frame):
        """Detect license plates in the given frame"""
        # Convert to grayscale for cascade classifier
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect plates
        plates = self.plate_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        # Return the plate regions
        return plates

    @staticmethod
    def draw_plates(frame, plates):
        """Draw rectangles around detected plates"""
        result = frame.copy()
        for (x, y, w, h) in plates:
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return result
    
    @staticmethod
    def extract_plate_regions(frame, plates):
        """Extract the plate regions from the frame"""
        plate_regions = []
        for (x, y, w, h) in plates:
            plate_region = frame[y:y+h, x:x+w]
            plate_regions.append(plate_region)
        return plate_regions


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NumberPlateDetector (Pakistan) - By Abdul Samad")
        self.setGeometry(100, 100, 1000, 700)
        
        # Initialize parameters first
        self.scale_factor = 1.1  # Default scale factor
        self.min_neighbors = 5  # Default min neighbors
        
        # Create the detector
        try:
            self.detector = PlateDetector()
            self.detection_enabled = True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize plate detector: {e}")
            self.detection_enabled = False
        
        # Create UI
        self.setup_ui()
        
        # Video capture setup
        self.video_path = "outVideo.avi"  # Default video path
        self.image_path = None  # For still image processing
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Initialize state
        self.current_frame = None
        self.current_plates = []
        self.current_plate_imgs = []
        self.processing_image = False
    
    def setup_ui(self):
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)
        
        # Create Video tab
        video_tab = QWidget()
        video_layout = QVBoxLayout(video_tab)
        tab_widget.addTab(video_tab, "Video Processing")
        
        # Create Image tab
        image_tab = QWidget()
        image_layout = QVBoxLayout(image_tab)
        tab_widget.addTab(image_tab, "Image Processing")
        
        # === VIDEO TAB CONTENT ===
        
        # Main video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 1px solid black")
        video_layout.addWidget(self.video_label)
        
        # Video control buttons
        video_controls = QHBoxLayout()
        
        self.load_video_button = QPushButton("Load Video")
        self.load_video_button.clicked.connect(self.load_video)
        video_controls.addWidget(self.load_video_button)
        
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.start_video)
        video_controls.addWidget(self.play_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_video)
        video_controls.addWidget(self.stop_button)
        
        self.clear_video_button = QPushButton("Clear")
        self.clear_video_button.clicked.connect(self.clear_video_results)
        video_controls.addWidget(self.clear_video_button)
        
        video_layout.addLayout(video_controls)
        
        # Text display for video detection results
        self.video_text_display = QTextEdit()
        self.video_text_display.setMaximumHeight(100)
        self.video_text_display.setReadOnly(True)
        video_layout.addWidget(self.video_text_display)
        
        # Video plate thumbnails list
        self.video_plate_list = QListWidget()
        self.video_plate_list.setMaximumHeight(120)
        self.video_plate_list.setFlow(QListWidget.LeftToRight)
        video_layout.addWidget(self.video_plate_list)
        
        # === IMAGE TAB CONTENT ===
        
        # Main image display
        self.image_label = QLabel()
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid black")
        image_layout.addWidget(self.image_label)
        
        # Image control buttons
        image_controls = QHBoxLayout()
        
        self.load_image_button = QPushButton("Load Image")
        self.load_image_button.clicked.connect(self.load_image)
        image_controls.addWidget(self.load_image_button)
        
        self.detect_button = QPushButton("Detect Plates")
        self.detect_button.clicked.connect(self.process_image)
        image_controls.addWidget(self.detect_button)
        
        self.save_button = QPushButton("Save Result")
        self.save_button.clicked.connect(self.save_result)
        image_controls.addWidget(self.save_button)
        
        self.clear_image_button = QPushButton("Clear")
        self.clear_image_button.clicked.connect(self.clear_image_results)
        image_controls.addWidget(self.clear_image_button)
        
        image_layout.addLayout(image_controls)
        
        # Detector parameter controls
        param_layout = QHBoxLayout()
        
        param_layout.addWidget(QLabel("Scale Factor:"))
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setMinimum(110)
        self.scale_slider.setMaximum(150)
        self.scale_slider.setValue(int(self.scale_factor * 100))
        self.scale_slider.valueChanged.connect(self.update_scale_factor)
        param_layout.addWidget(self.scale_slider)
        
        param_layout.addWidget(QLabel("Min Neighbors:"))
        self.neighbors_slider = QSlider(Qt.Horizontal)
        self.neighbors_slider.setMinimum(1)
        self.neighbors_slider.setMaximum(10)
        self.neighbors_slider.setValue(self.min_neighbors)
        self.neighbors_slider.valueChanged.connect(self.update_min_neighbors)
        param_layout.addWidget(self.neighbors_slider)
        
        image_layout.addLayout(param_layout)
        
        # Text display for image detection results
        self.image_text_display = QTextEdit()
        self.image_text_display.setMaximumHeight(100)
        self.image_text_display.setReadOnly(True)
        image_layout.addWidget(self.image_text_display)
        
        # Image plate thumbnails list
        self.image_plate_list = QListWidget()
        self.image_plate_list.setMaximumHeight(120)
        self.image_plate_list.setFlow(QListWidget.LeftToRight)
        image_layout.addWidget(self.image_plate_list)
    
    def update_scale_factor(self, value):
        self.scale_factor = value / 100.0
        self.image_text_display.setText(f"Scale Factor: {self.scale_factor}, Min Neighbors: {self.min_neighbors}")
    
    def update_min_neighbors(self, value):
        self.min_neighbors = value
        self.image_text_display.setText(f"Scale Factor: {self.scale_factor}, Min Neighbors: {self.min_neighbors}")
    
    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv)"
        )
        
        if file_path:
            self.video_path = file_path
            self.video_text_display.setText(f"Loaded video: {os.path.basename(file_path)}")
            self.stop_video()  # Stop any running video
            self.clear_video_results()
    
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if file_path:
            self.image_path = file_path
            img = cv2.imread(file_path)
            if img is not None:
                self.current_frame = img.copy()
                self.display_image(img, self.image_label)
                self.image_text_display.setText(f"Loaded image: {os.path.basename(file_path)}")
                self.clear_image_results()
            else:
                self.image_text_display.setText(f"Error loading image: {os.path.basename(file_path)}")
    
    def process_image(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "Warning", "Please load an image first")
            return
        
        # Create a copy of the current frame
        img = self.current_frame.copy()
        
        # Detect plates with updated parameters
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB if len(img.shape) > 2 else img)
        plates = self.detector.plate_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(30, 30)
        )
        
        # Extract plate regions
        plate_regions = self.detector.extract_plate_regions(img, plates)
        
        # Draw plates on image
        result_img = self.detector.draw_plates(img, plates)
        
        # Display the result
        self.display_image(result_img, self.image_label)
        
        # Update the plate count text
        self.image_text_display.setText(f"Detected {len(plates)} license plates")
        
        # Update plate thumbnails
        self.image_plate_list.clear()
        for img in plate_regions:
            # Add thumbnail to list
            thumbnail = self.convert_cv_to_pixmap(img)
            if thumbnail:
                item_label = QLabel()
                item_label.setPixmap(thumbnail.scaled(100, 60, Qt.KeepAspectRatio))
                
                # Add to list
                item = QListWidgetItem()
                item.setSizeHint(item_label.sizeHint())
                self.image_plate_list.addItem(item)
                self.image_plate_list.setItemWidget(item, item_label)
    
    def save_result(self):
        if self.image_label.pixmap() is None:
            QMessageBox.warning(self, "Warning", "No image to save")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "Image Files (*.jpg *.png)"
        )
        
        if file_path:
            pixmap = self.image_label.pixmap()
            pixmap.save(file_path)
            self.image_text_display.setText(f"Image saved as: {os.path.basename(file_path)}")
    
    def start_video(self):
        if self.cap is None or not self.cap.isOpened():
            try:
                self.cap = cv2.VideoCapture(self.video_path)
                if not self.cap.isOpened():
                    QMessageBox.warning(self, "Warning", f"Failed to open video file: {self.video_path}")
                    return
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error opening video: {e}")
                return
        
        self.timer.start(30)  # Update every 30ms (~33 fps)
        self.video_text_display.setText("Video playing...")
    
    def stop_video(self):
        self.timer.stop()
        self.video_text_display.setText("Video stopped")
    
    def clear_video_results(self):
        self.video_plate_list.clear()
        self.video_text_display.clear()
        self.current_plates = []
        self.current_plate_imgs = []
    
    def clear_image_results(self):
        self.image_plate_list.clear()
        if self.current_frame is not None:
            self.display_image(self.current_frame, self.image_label)
    
    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return
        
        ret, frame = self.cap.read()
        if not ret:
            # Restart video at the end
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return
        
        # Resize for processing
        frame = cv2.resize(frame, (640, 480))
        
        # Detect license plates
        if self.detection_enabled:
            self.current_plates = self.detector.detect_plates(frame)
            
            # Extract plate regions
            self.current_plate_imgs = self.detector.extract_plate_regions(frame, self.current_plates)
            
            # Draw plates on frame
            frame = self.detector.draw_plates(frame, self.current_plates)
            
            # Update the plate count text
            self.video_text_display.setText(f"Detected {len(self.current_plates)} license plates")
            
            # Update plate thumbnails (only for new detections)
            if len(self.current_plate_imgs) > self.video_plate_list.count():
                for img in self.current_plate_imgs[self.video_plate_list.count():]:
                    # Add thumbnail to list
                    thumbnail = self.convert_cv_to_pixmap(img)
                    if thumbnail:
                        item_label = QLabel()
                        item_label.setPixmap(thumbnail.scaled(100, 60, Qt.KeepAspectRatio))
                        
                        # Add to list
                        item = QListWidgetItem()
                        item.setSizeHint(item_label.sizeHint())
                        self.video_plate_list.addItem(item)
                        self.video_plate_list.setItemWidget(item, item_label)
        
        # Display the frame
        self.display_frame(frame)
    
    def convert_cv_to_pixmap(self, cv_img):
        """Convert OpenCV image to QPixmap"""
        if cv_img is None or cv_img.size == 0:
            return None
            
        # Convert to RGB format for Qt
        try:
            if len(cv_img.shape) == 3:  # Color image
                rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            else:  # Grayscale image
                rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
                
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            
            # Create QImage and QPixmap
            q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            return QPixmap.fromImage(q_img)
        except Exception as e:
            print(f"Error converting image: {e}")
            return None
    
    def display_frame(self, frame):
        """Display the frame in the video label"""
        pixmap = self.convert_cv_to_pixmap(frame)
        if pixmap:
            self.video_label.setPixmap(pixmap.scaled(
                self.video_label.width(),
                self.video_label.height(),
                Qt.KeepAspectRatio
            ))
    
    def display_image(self, image, label):
        """Display an image in the specified label"""
        pixmap = self.convert_cv_to_pixmap(image)
        if pixmap:
            label.setPixmap(pixmap.scaled(
                label.width(),
                label.height(),
                Qt.KeepAspectRatio
            ))
    
    def closeEvent(self, event):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    
    # Set style to Fusion for a consistent look
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error running application: {e}") 