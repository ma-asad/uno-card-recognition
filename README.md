
# UNO Card Detector

A Python application for detecting UNO cards using a neural network model. This project includes a GUI that allows users to upload images or use a live camera feed to detect UNO cards in real-time.

This project is part of the module **AI in Robotics (PDE3821) coursework 1** of **Middlesex University**.

**Authors**: Muhammad Ayman Seeraullee, Mohammad Asad Atterkhan, Bibi Hafsah Joomun

## Table of Contents

- Features
- Installation
- Usage
  - Running the Application
  - Using the GUI
- Model Training
- Dependencies

## Features

- Detects UNO cards from images or live camera feed.
- Provides a user-friendly GUI built with `ttkbootstrap`.
- Loads a pre-trained neural network model for card classification.
- Supports a wide range of UNO card types, including numeric and action cards.

## Installation

### Prerequisites

- Python 3.10 or later
- [PyTorch](https://pytorch.org/) for neural network functionality
- [ttkbootstrap](https://github.com/israel-dryer/ttkbootstrap) for the GUI
- Other dependencies listed in `requirements.txt`

### Steps

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Download the Pre-trained Model**

   Ensure that the `full_symbol_classifier.pth` file is in the project directory. This file contains the pre-trained neural network model.

## Usage

### Running the Application

Start the GUI application by running:

```bash
python main.py
```

### Using the GUI

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

## Model Training

The neural network model is defined in `uno_model.ipynb`. To train the model:

1. **Prepare the Dataset**

   - Collect and label images of UNO cards.
   - Organize the data into training, validation, and testing sets.

2. **Train the Model**

   - Open `uno_model.ipynb` in Jupyter Notebook.
   - Run the notebook cells to train the model.
   - Adjust hyperparameters as needed.

3. **Save the Model**

   - After training, save the model as `full_symbol_classifier.pth`.
   - Place this file in the project directory.

## Dependencies

- Python 3.10
- PyTorch
- Torchvision
- ttkbootstrap
- Pillow
- OpenCV
- Tkinter

Install all dependencies using:

```bash
pip install -r requirements.txt
```

