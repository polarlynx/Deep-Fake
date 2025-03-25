# Deep-Fake
Eliminate image Ai traces and separate layers to make them editable.

## Main Features
1. Generate line art  
2. Color segmentation  
3. Mask generation  
4. PSD packaging  

## Environment
- **Python version**: 3.10.16  
- Install dependencies:
  ```
  pip install -r requirements.txt
  ```

## How to Run
1. **Directly with .exe**  
   - Download `Deepfake.exe` from Releases and run it.  
2. **Using Python**  
   - Make sure you have Python 3.10.16  
   - Run `python main.py`

## Usage
1. Select the source image.  
2. Adjust parameters (max images, min area, kernel size, etc.).  
3. Click **Run Processing**.  
4. Wait for the process to complete.

## Files Generated
- **source** folder: Segmented images  
- **masked** folder: Generated masks  
- **final.psd**: Final PSD file with all layers  

Customize parameters or scripts as needed, have fun!
