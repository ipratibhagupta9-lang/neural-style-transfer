# Neural Style Transfer 

## Overview

Neural Style Transfer Web App is a simple AI application that transforms an image by applying the artistic style of another image. The project uses Google's TensorFlow Hub Arbitrary Image Stylization model and provides an easy-to-use interface built with Streamlit.

## Features

* Upload a content image and a style image
* Generate artistic images using AI
* Light and Dark theme support
* Image preview before processing
* Download the stylized image
* Simple and responsive user interface

## Technologies Used

* Python
* Streamlit
* TensorFlow
* TensorFlow Hub
* NumPy
* Pillow

## Run the Application

Start the application using:

```bash
streamlit run neural_style_transfer1.py
```

After running the command, open the local URL displayed in the terminal.

## How to Use

1. Upload a content image.
2. Upload a style image.
3. Click **Generate Stylized Image**.
4. Wait for the AI model to process the images.
5. View and download the final stylized image.

## Project Structure

```text
neural-style-transfer-webapp/
│── neural_style_transfer1.py
│── Screenshot1
│── Screenshot2
└── README.md
```

## Future Enhancements

* Support multiple artistic styles
* Style intensity adjustment
* Higher-resolution output
* Faster image processing
* More AI style transfer models
