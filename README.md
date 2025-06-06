# Microscope Calibration Tool

A Python-based tool for calibrating electron microscopes by analyzing test specimen images. This tool helps calculate the pixel size (Angstroms/pixel) by measuring diffraction patterns from known specimens like graphene, gold, or ice.

## Features

- Support for multiple image formats:
  - Common formats (.png, .jpg, .tif)
  - MRC files from microscopes
- Interactive FFT analysis with resolution circles
- Automatic pixel size detection
- Radial averaging for enhanced signal detection
- Customizable resolution measurements
- Real-time visualization and analysis

## Test Data

The `test_image` folder contains a sample image for testing the application:
- A graphene specimen image collected at nominal magnification of 0.75
- Image was collected at the Cryo-Electron Microscopy Facility at Penn State University.
- Can be used to verify the application's functionality and calibration process
- Ideal for first-time users to familiarize themselves with the tool

## Installation

1. Clone this repository:
```bash
git clone https://github.com/jianglab/magCalApp.git
cd magCalApp
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- shiny>=0.6.0
- Pillow>=10.0.0
- numpy>=1.24.0
- matplotlib>=3.7.0
- mrcfile>=1.4.0

## Usage

1. Start the application:
```bash
python app.py
```

2. Using the web interface:
   - Upload a test specimen image
   - Select the expected diffraction pattern (graphene/gold/ice)
   - Adjust the region size to analyze
   - Click points in the FFT to measure distances
   - Use auto-search to find the best pixel size match

## Interface Components

### Sidebar Controls
- File upload for test specimen images
- Toggle buttons for different specimen types:
  - Graphene (2.13 Å)
  - Gold (2.355 Å)
  - Ice (3.661 Å)
- Custom resolution input
- Apix slider (0.01-6.0 Å/px)
- Auto-search range controls

### Main Display
1. **Original Image View**
   - Shows uploaded image with selected FFT region
   - Interactive region selection
   - Zoom and contrast controls
   
2. **FFT Display**
   - Shows FFT of selected region
   - Resolution circles based on selected specimens
   - Interactive calibration through clicking
   
3. **Radial Profile Plot**
   - 1D rotational average of the FFT
   - Resolution markers
   - Zoom and pan capabilities
   - Toggle between linear and log scale

## Analysis Features

### Region Selection
- Click on the original image to select the feature of interests with known resolution (graphene, gold, ice,...)
- Adjustable region size using slider
- Real-time FFT update

### Calibration Methods
1. **Manual Calibration**
   - Click on FFT features to match with known spacings
   - Real-time Apix calculation
   
2. **Auto-Local Search**
   - Automatic detection of local peak.
   - Searches specified small Apix range.
   - Finds best match for selected spacing

### Visualization Controls
- Contrast adjustment for both original and FFT images
- Zoom controls for detailed inspection
- Interactive plot with zoom and pan
- Resolution circle overlays

## Tips for Best Results

1. Use high-quality test specimens
2. Ensure proper specimen orientation
3. Start with approximate Apix value if known
4. Use auto-search feature to refine manual measurements
5. Compare results across different regions of the image

## Troubleshooting

- If FFT appears noisy:
  - Increase region size
  - Adjust contrast
  - Try different image areas
- If auto-search fails:
  - Narrow the Apix search range
  - Verify specimen selection
  - Check image quality

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
