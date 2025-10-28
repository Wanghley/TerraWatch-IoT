#!/usr/bin/env python3
"""
TerraWatch IoT L3 Script
Takes an input image from the data folder and runs YOLOv5 inference
"""

import os
import sys
import subprocess
import shutil
from datetime import datetime
from pathlib import Path

def capture_camera_frame():
    """
    Capture a frame from the camera using ffmpeg
    
    Returns:
        str: Path to the captured image file, or None if failed
    """
    output_path = Path("/home/orangepi/TerraWatch-IoT/firmware-sensing-l3/output.jpg")
    
    print("Capturing frame from camera...")
    print("Command: ffmpeg -f video4linux2 -i /dev/video1 -vframes 1 output.jpg -y")
    
    try:
        result = subprocess.run([
            "ffmpeg", "-f", "video4linux2", "-i", "/dev/video1", 
            "-vframes", "1", str(output_path), "-y"
        ], capture_output=True, text=True, check=True)
        
        if output_path.exists():
            print(f"Camera frame captured successfully: {output_path}")
            return str(output_path)
        else:
            print("Error: Camera frame capture failed - output file not created")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Error capturing camera frame: {e}")
        print(f"Return code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        return None
    except Exception as e:
        print(f"Unexpected error capturing camera frame: {e}")
        return None

def run_yolov5_inference(input_image):
    """
    Run YOLOv5 inference on the specified input image
    
    Args:
        input_image (str): Name of the input image file (e.g., 'dog.jpg')
    """
    
    # Define paths
    data_folder = Path("/home/orangepi/TerraWatch-IoT/firmware-sensing-l3/data")
    yolov5_build_dir = Path("/home/orangepi/TerraWatch-IoT/board-demo/yolov5/build")
    results_folder = Path("/home/orangepi/TerraWatch-IoT/firmware-sensing-l3/results")
    model_path = "../model/v2/yolov5.nb"
    
    # Generate timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Check if input image exists (try data folder first, then main directory)
    input_image_path = data_folder / input_image
    if not input_image_path.exists():
        # Try main directory (for camera-captured images)
        main_dir_path = Path("/home/orangepi/TerraWatch-IoT/firmware-sensing-l3") / input_image
        if main_dir_path.exists():
            input_image_path = main_dir_path
        else:
            print(f"Error: Input image '{input_image}' not found in {data_folder} or main directory")
            return False
    
    # Check if YOLOv5 executable exists
    yolov5_executable = yolov5_build_dir / "yolov5"
    if not yolov5_executable.exists():
        print(f"Error: YOLOv5 executable not found at {yolov5_executable}")
        return False
    
    # Check if model file exists
    model_file_path = yolov5_build_dir / model_path
    if not model_file_path.exists():
        print(f"Error: Model file not found at {model_file_path}")
        return False
    
    # Ensure the image is available in yolov5's expected ../input_data directory
    input_data_dir = (yolov5_build_dir / ".." / "input_data").resolve()
    try:
        input_data_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error ensuring input_data directory exists at {input_data_dir}: {e}")
        return False

    # Copy source image into input_data with same filename so relative path works
    input_data_image_path = input_data_dir / input_image
    try:
        shutil.copy2(input_image_path, input_data_image_path)
    except Exception as e:
        print(f"Error copying image to input_data: {e}")
        return False

    # Prepare the command
    input_data_path = f"../input_data/{input_image}"
    command = ["./yolov5", model_path, input_data_path]
    
    print(f"Running YOLOv5 inference on: {input_image}")
    print(f"Command: {' '.join(command)}")
    print(f"Working directory: {yolov5_build_dir}")
    
    try:
        # Change to YOLOv5 build directory and run the command
        result = subprocess.run(
            command,
            cwd=yolov5_build_dir,
            capture_output=True,
            text=True,
            check=True
        )
        
        print("YOLOv5 inference completed successfully!")
        print("Output:")
        print(result.stdout)
        
        if result.stderr:
            print("Warnings/Info:")
            print(result.stderr)
        
        # Save console output to timestamped file
        output_file = results_folder / f"result_{timestamp}.txt"
        with open(output_file, 'w') as f:
            f.write(f"YOLOv5 Inference Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input Image: {input_image}\n")
            f.write(f"Command: {' '.join(command)}\n")
            f.write(f"Working Directory: {yolov5_build_dir}\n")
            f.write("="*50 + "\n\n")
            f.write("STDOUT:\n")
            f.write(result.stdout)
            if result.stderr:
                f.write("\nSTDERR:\n")
                f.write(result.stderr)
        
        print(f"Console output saved to: {output_file}")
        
        # Copy result.png to results folder with timestamp
        result_png_source = yolov5_build_dir / "result.png"
        if result_png_source.exists():
            result_png_dest = results_folder / f"result_{timestamp}.png"
            shutil.copy2(result_png_source, result_png_dest)
            print(f"Result image copied to: {result_png_dest}")
        else:
            print("Warning: result.png not found in YOLOv5 build directory")
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error running YOLOv5: {e}")
        print(f"Return code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        
        # Save error output to timestamped file
        error_file = results_folder / f"error_{timestamp}.txt"
        with open(error_file, 'w') as f:
            f.write(f"YOLOv5 Error Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input Image: {input_image}\n")
            f.write(f"Command: {' '.join(command)}\n")
            f.write(f"Working Directory: {yolov5_build_dir}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Error: {e}\n")
            f.write(f"Return code: {e.returncode}\n")
            f.write(f"Error output: {e.stderr}\n")
        
        print(f"Error log saved to: {error_file}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        
        # Save unexpected error to timestamped file
        error_file = results_folder / f"error_{timestamp}.txt"
        with open(error_file, 'w') as f:
            f.write(f"Unexpected Error Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input Image: {input_image}\n")
            f.write(f"Command: {' '.join(command)}\n")
            f.write(f"Working Directory: {yolov5_build_dir}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Unexpected error: {e}\n")
        
        print(f"Error log saved to: {error_file}")
        return False

def main():
    """Main function to handle command line arguments"""
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 l3.py <INPUT_IMAGE>                    # Run inference on image from data folder")
        print("  python3 l3.py --camera-input                   # Capture frame from camera and run inference")
        print("\nExamples:")
        print("  python3 l3.py dog.jpg")
        print("  python3 l3.py --camera-input")
        print("\nAvailable images in data folder:")
        
        # List available images
        data_folder = Path("/home/orangepi/TerraWatch-IoT/firmware-sensing-l3/data")
        if data_folder.exists():
            image_files = [f for f in data_folder.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            for img in image_files:
                print(f"  - {img.name}")
        else:
            print("  Data folder not found")
        
        sys.exit(1)
    
    # Check if camera input flag is used
    if sys.argv[1] == "--camera-input":
        print("Camera input mode selected")
        
        # Capture frame from camera
        camera_image_path = capture_camera_frame()
        if camera_image_path is None:
            print("Failed to capture camera frame. Exiting.")
            sys.exit(1)
        
        # Use the captured image for inference
        input_image = Path(camera_image_path).name
        print(f"Running inference on captured camera frame: {input_image}")
        
        # Run the inference
        success = run_yolov5_inference(input_image)
        
    else:
        # Regular image file input
        input_image = sys.argv[1]
        
        # Run the inference
        success = run_yolov5_inference(input_image)
    
    if success:
        print("\nInference completed successfully!")
        sys.exit(0)
    else:
        print("\nInference failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
