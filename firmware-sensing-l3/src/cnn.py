#!/usr/bin/env python3
"""
CNN inference script for capturing frames, running YOLOv5 inference, and saving results.
"""

import subprocess
import shutil
from datetime import datetime
from pathlib import Path

# Define paths
WORKSPACE_ROOT = Path(__file__).parent.parent.parent
FIRMWARE_ROOT = WORKSPACE_ROOT / "firmware-sensing-l3"
BOARD_DEMO_ROOT = WORKSPACE_ROOT / "board-demo" / "yolov5"

# Input/Output paths
CAPTURE_OUTPUT = FIRMWARE_ROOT / "src" / "output.jpg"
YOLO_INPUT = BOARD_DEMO_ROOT / "input_data" / "output.jpg"
YOLO_BUILD_DIR = BOARD_DEMO_ROOT / "build"
YOLO_MODEL = BOARD_DEMO_ROOT / "model" / "v2" / "yolov5.nb"
YOLO_RESULT = YOLO_BUILD_DIR / "result.png"
RESULTS_DIR = FIRMWARE_ROOT / "results"

# Video device
VIDEO_DEVICE = "/dev/video1"


def capture_frame():
    """Capture a frame from the video device using ffmpeg."""
    print("Capturing frame from video device...")
    cmd = [
        "ffmpeg",
        "-f", "video4linux2",
        "-i", VIDEO_DEVICE,
        "-vframes", "1",
        "-y",  # Overwrite output file
        "-t", "1",  # Limit capture time to 1 second
        str(CAPTURE_OUTPUT)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=5  # 5 second timeout to prevent hanging
        )
        print(f"Frame captured successfully: {CAPTURE_OUTPUT}")
        return True
    except subprocess.TimeoutExpired:
        print("Error: Frame capture timed out after 5 seconds")
        return False
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        print(f"Error capturing frame: {error_msg}")
        return False


def copy_frame_to_yolo_input():
    """Copy the captured frame to YOLO input directory."""
    print(f"Copying frame to YOLO input directory...")
    try:
        # Ensure input_data directory exists
        YOLO_INPUT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CAPTURE_OUTPUT, YOLO_INPUT)
        print(f"Frame copied to: {YOLO_INPUT}")
        return True
    except Exception as e:
        print(f"Error copying frame: {e}")
        return False


def run_yolo_inference():
    """Run YOLOv5 inference and capture detection labels from stderr."""
    print("Running YOLOv5 inference...")
    
    # Change to build directory and run yolo
    # Using relative paths as specified: ../model/v2/yolov5.nb ../input_data/output.jpg
    cmd = [
        "./yolov5",
        "../model/v2/yolov5.nb",
        "../input_data/output.jpg"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(YOLO_BUILD_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        
        # Parse stderr for detection labels
        stderr_lines = result.stderr.strip().split('\n')
        detection_lines = []
        
        for line in stderr_lines:
            # Skip the "detection num: X" line and empty lines
            if line.strip() and not line.startswith("detection num:"):
                # Check if line matches the detection format: "16:  91%, [ 135,  218,  305,  553], dog"
                if ':' in line and '%' in line and '[' in line:
                    detection_lines.append(line.strip())
        
        print(f"Found {len(detection_lines)} detections")
        return detection_lines, True
    except subprocess.CalledProcessError as e:
        print(f"Error running YOLO inference: {e.stderr}")
        return [], False


def save_results(detection_lines):
    """Save result image and labels to results directory with timestamp."""
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Copy result image
    result_image = RESULTS_DIR / f"result_{timestamp}.png"
    try:
        if YOLO_RESULT.exists():
            shutil.copy2(YOLO_RESULT, result_image)
            print(f"Result image saved: {result_image}")
        else:
            print(f"Warning: Result image not found at {YOLO_RESULT}")
    except Exception as e:
        print(f"Error copying result image: {e}")
    
    # Save detection labels
    result_labels = RESULTS_DIR / f"result_{timestamp}.txt"
    try:
        with open(result_labels, 'w') as f:
            for line in detection_lines:
                f.write(line + '\n')
        print(f"Detection labels saved: {result_labels}")
        return True
    except Exception as e:
        print(f"Error saving detection labels: {e}")
        return False


def main():
    """Main execution function."""
    print("=" * 60)
    print("YOLOv5 Inference Pipeline")
    print("=" * 60)
    
    # Step 1: Capture frame
    if not capture_frame():
        print("Failed to capture frame. Exiting.")
        return
    
    # Step 2: Copy frame to YOLO input directory
    if not copy_frame_to_yolo_input():
        print("Failed to copy frame. Exiting.")
        return
    
    # Step 3: Run YOLO inference
    detection_lines, success = run_yolo_inference()
    if not success:
        print("Failed to run YOLO inference. Exiting.")
        return
    
    # Step 4 & 5: Save results
    save_results(detection_lines)
    
    print("=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
