# Kannada Handwritten Text Recognition

This project uses Optical Character Recognition (OCR) to extract handwritten text from images in Kannada script.

## Features

- Upload images in various formats (PNG, JPG, JPEG).
- Multiple OCR methods:
  - *Tesseract*
  - *EasyOCR*
  - *TrOCR*
- Image preprocessing options.

## Requirements

- Python 3.7 or higher
- Required Python packages (listed in requirements.txt)

## Installation

1.
Create a virtual environment:

bash

python -m venv venv
Install the required packages:

bash
pip install -r requirements.txt

Usage
Start the Streamlit application:

bash

streamlit run app.py
Open your web browser at http://localhost:8501.

Upload an image and select an OCR method to extract text.

Code Documentation
Ensure that all code submissions include sufficient inline documentation to explain the functionality and logic of the code. This should include:

Function and method descriptions.
Parameter explanations.
Return values and exceptions raised.
License
This project is licensed under the GNU General Public License v3.0 (GPLv3).
