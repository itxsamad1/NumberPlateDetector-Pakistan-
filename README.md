# Pakistani Number Plate Detector 🚗🇵🇰

A computer vision application designed specifically for detecting and recognizing license plates from Pakistani vehicles using advanced image processing techniques.

![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![PyQt5](https://img.shields.io/badge/PyQt5-GUI-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🚀 Features

- **Video Processing**: Load and analyze video files to detect license plates in real-time
- **Image Processing**: Process static images for license plate detection
- **Adjustable Parameters**: Fine-tune detection with scale factor and minimum neighbors sliders
- **Result Management**: Save detected plate images and view detection results
- **User-Friendly Interface**: Intuitive tabbed interface for easy navigation
- **Pakistani Plate Optimization**: Specifically trained for Pakistani license plate formats
- **Batch Processing**: Process multiple frames/images efficiently

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.6+**: Main programming language
- **OpenCV**: Computer vision and image processing
- **PyQt5**: GUI framework for desktop application
- **NumPy**: Numerical computing (implicit dependency)

### Machine Learning
- **Haar Cascade Classifier**: Custom trained classifier (`pak.xml`) for Pakistani license plates
- **Computer Vision**: Image preprocessing and detection algorithms

### File Formats Supported
- **Video**: `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`
- **Images**: `.jpg`, `.jpeg`, `.png`, `.bmp`

## 📋 Prerequisites

- Python 3.6 or higher
- Windows/Linux/macOS
- Webcam (optional, for real-time detection)

## 🚀 Quick Start

### Option 1: Easy Installation (Windows)
Simply run the provided batch file:
```bash
run.bat
```

This will automatically:
- Check if Python is installed
- Install required packages
- Verify required files exist
- Launch the application

### Option 2: Manual Installation

1. **Clone the repository**
```bash
git clone https://github.com/itxsamad1/NumberPlateDetector-Pakistan-.git
cd NumberPlateDetector-Pakistan-
```

2. **Install dependencies**
```bash
pip install opencv-python PyQt5 numpy
```

3. **Run the application**
```bash
python main.py
```

## 📖 Usage Guide

### Video Processing
1. Click on **"Load Video"** button
2. Select a video file from supported formats
3. The video will start playing in the main window
4. Detected license plates will appear in the right panel
5. Click **"Save Results"** to save all detected plates

### Image Processing
1. Click on **"Load Image"** button
2. Select an image file from supported formats
3. The image will display in the main window
4. Adjust detection parameters using sliders:
   - **Scale Factor**: Controls detection window scale step size
   - **Min Neighbors**: Controls neighbor validation for detection
5. Click **"Detect Plates"** to run detection
6. View results in the right panel
7. Click **"Save Results"** to save detected plates

### Additional Features
- **Clear Results**: Remove all detected plates from current session
- **Tab Navigation**: Switch between video and image processing modes
- **Real-time Adjustment**: Modify detection parameters on-the-fly

## 🏗️ Project Structure

```
NumberPlateDetector-Pakistan-/
├── main.py                 # Main application entry point
├── pak.xml                 # Haar Cascade classifier for Pakistani plates
├── run.bat                 # Windows batch file for easy setup
├── requirements.txt        # Python dependencies
├── assets/                 # Application assets and icons
├── results/               # Saved detection results
└── README.md              # Project documentation
```

## ⚙️ Configuration

### Detection Parameters
- **Scale Factor** (1.05 - 1.3): Lower values = more thorough detection, higher computational cost
- **Min Neighbors** (3 - 10): Higher values = fewer false positives, might miss some plates

### Custom Training
The application uses a custom Haar Cascade classifier (`pak.xml`) specifically trained for Pakistani license plates. This ensures better accuracy for local plate formats and fonts.

## 🎯 Performance Tips

- Use scale factor between 1.1-1.2 for optimal results
- Set min neighbors to 4-6 for balanced accuracy
- Ensure good lighting conditions for better detection
- Clean, high-resolution images yield better results

## 🐛 Troubleshooting

### Common Issues
1. **Application won't start**: Ensure Python 3.6+ is installed
2. **Poor detection**: Adjust scale factor and min neighbors parameters
3. **Missing dependencies**: Run `pip install -r requirements.txt`
4. **Video won't load**: Check if video format is supported

### Performance Issues
- Close other applications to free up system resources
- Use smaller video files for faster processing
- Reduce video resolution if processing is slow

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**itxsamad1**
- GitHub: [@itxsamad1](https://github.com/itxsamad1)

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- PyQt5 developers for the GUI framework
- Contributors to Haar Cascade training techniques
- Pakistani automotive community for license plate samples

## 📈 Future Enhancements

- [ ] OCR integration for text extraction
- [ ] Real-time webcam detection
- [ ] Mobile app version
- [ ] Multiple plate format support
- [ ] Database integration for plate lookup
- [ ] Advanced neural network models

## 📞 Support

If you encounter any issues or have questions, please:
1. Check the troubleshooting section
2. Open an issue on GitHub
3. Contact the maintainer

---

⭐ **If this project helped you, please give it a star!** ⭐
