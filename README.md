# Deep-Fake
Eliminate image Ai traces and separate layers to make them editable.

## Main Features
1. Generate line art  
2. Color segmentation  
3. Mask generation  
4. PSD packaging  

## Environment Setup
Run:
```
pip install -r requirements.txt
```
to install all required dependencies.

## Usage
1. Execute `python main.py`.
2. Click **Select Image** in the GUI to choose an image.
3. Adjust parameters such as max images, min area, kernel size, etc.
4. Click **Run Processing** and wait for the process to finish.

## File Description
- **source** folder: contains the separately segmented images  
- **masked** folder: contains the generated masks  
- **final.psd**: the final layered PSD file  

Customize parameters or scripts as needed.  
