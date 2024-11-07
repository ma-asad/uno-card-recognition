# UNO Card Detector

A Python application for detecting UNO cards using two different methods: a neural network model and an ORB-based feature detection method. This project includes a GUI for the CNN-based system and a web interface for the ORB-based system, allowing users to upload images or use a live camera feed to detect UNO cards in real-time.

This project is part of the module **AI in Robotics (PDE3821) coursework 1** of **Middlesex University**.

**Authors**:
- Mohammad Asad Atterkhan
- Bibi Hafsah Joomun
- Muhammad Ayman Seeraullee

## Table of Contents

- Features
- Installation
- Usage
  - Running the CNN-based Application
  - Running the ORB-based Application
  - Using the GUI
  - Using the Web Interface
- Model Training
- Dependencies

## Features

- Detects UNO cards from images or live camera feed.
- Provides a user-friendly GUI built with `ttkbootstrap` for the CNN-based system.
- Provides a web interface built with Flask for the ORB-based system.
- Loads a pre-trained neural network model for card classification in the CNN-based system.
- Uses ORB-based feature detection for card identification in the ORB-based system.
- Supports a wide range of UNO card types, including numeric and action cards.

## Installation

### Prerequisites

- Python 3.10 or later
- [PyTorch](https://pytorch.org/) for neural network functionality (CNN-based system)
- [ttkbootstrap](https://github.com/israel-dryer/ttkbootstrap) for the GUI (CNN-based system)
- [Flask](https://flask.palletsprojects.com/) for the web interface (ORB-based system)
- Other dependencies listed in `requirements.txt`

### Steps

1. **Install Dependencies**

   ```bash
   pip install -r cnn/requirements.txt
   pip install -r orb/requirements.txt
   ```

2. **Download the Pre-trained Model (for CNN-based system)**

   Ensure that the `full_symbol_classifier.pth` file is in the `cnn/src/data` directory. This file contains the pre-trained neural network model.

## Usage

### Running the CNN-based Application

Start the GUI application by running:

```bash
python cnn/src/main.py
```

### Running the ORB-based Application

Start the web application by running:

```bash
python orb/src/main.py
```

### Using the GUI (CNN-based system)

1. **Main Screen**

   - **Upload Image**: Allows you to select an image file containing UNO cards.
   - **Use Live Camera**: Activates your webcam to detect UNO cards in real-time.

2. **Uploading an Image**

   - Click on **Upload Image**.
   - Select an image file (`.jpg`, `.jpeg`, `.png`).
   - The application will display the image and show the detected card.

3. **Live Camera Detection**

   - Click on **Use Live Camera**.
   - The live camera feed will appear.
   - Detected cards will be displayed with their names.
   - Click **Back to Home** to return to the main screen.

### Using the Web Interface (ORB-based system)

1. **Main Page**

   - **Upload Image**: Allows you to select an image file containing UNO cards.
   - **Use Live Camera**: Activates your webcam to detect UNO cards in real-time.

2. **Uploading an Image**

   - Click on **Upload Image**.
   - Select an image file (`.jpg`, `.jpeg`, `.png`).
   - The application will display the image and show the detected card.

3. **Live Camera Detection**

   - Click on **Use Live Camera**.
   - The live camera feed will appear.
   - Detected cards will be displayed with their names.

## Model Training (for CNN-based system)

The neural network model is defined in `cnn/uno_model.ipynb`. To train the model:

1. **Prepare the Dataset**

   - Collect and label images of UNO cards.
   - Organize the data into training, validation, and testing sets.

2. **Train the Model**

   - Open `cnn/uno_model.ipynb` in Jupyter Notebook.
   - Run the notebook cells to train the model.
   - Adjust hyperparameters as needed.

3. **Save the Model**

   - After training, save the model as `full_symbol_classifier.pth`.
   - Place this file in the `cnn/src/data` directory.

## Dependencies

- Python 3.10
- PyTorch (for CNN-based system)
- Torchvision (for CNN-based system)
- ttkbootstrap (for CNN-based system)
- Pillow
- OpenCV
- Tkinter (for CNN-based system)
- Flask (for ORB-based system)
- Flask-CORS (for ORB-based system)

Install all dependencies using:

```bash
pip install -r cnn/requirements.txt
pip install -r orb/requirements.txt
```
