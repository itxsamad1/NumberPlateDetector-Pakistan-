# NumberPlateDetector (Pakistan)
## By Abdul Samad

This application detects and recognizes license plates from Pakistani vehicles using computer vision techniques.

## Features

- **Video Processing**: Load and analyze video files to detect license plates
- **Image Processing**: Load and analyze image files to detect license plates
- **Adjustable Parameters**: Fine-tune detection with scale factor and minimum neighbors sliders
- **Result Management**: Save detected plate images and view results
- **User-Friendly Interface**: Tabbed interface for easy navigation between video and image processing

## Requirements

- Python 3.6 or higher
- OpenCV
- PyQt5

## Installation

Simply run the `run.bat` file which will:
1. Check if Python is installed
2. Install required packages
3. Verify required files exist
4. Launch the application

## Usage

### Video Processing
1. Click on the "Load Video" button to select a video file (.mp4, .avi, .mov, .mkv, .wmv)
2. The video will play in the main window
3. Detected license plates will appear in the right panel
4. Click "Save Results" to save all detected plates

### Image Processing
1. Click on the "Load Image" button to select an image file (.jpg, .jpeg, .png, .bmp)
2. The image will display in the main window
3. Adjust detection parameters using the sliders if needed:
   - Scale Factor: Controls the step size of the detection window scale
   - Min Neighbors: Controls how many neighbors each candidate rectangle should have
4. Click "Detect Plates" to run the detection algorithm
5. Detected license plates will appear in the right panel
6. Click "Save Results" to save all detected plates

### Additional Controls
- "Clear Results" button removes all detected plates
- Switch between tabs to process either videos or images

## License

This project uses the Haar Cascade classifier (`pak.xml`) for license plate detection. 