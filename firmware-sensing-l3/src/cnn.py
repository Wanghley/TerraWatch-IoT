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
    """Capture a frame from the video device using ffmpeg.
    Captures multiple frames and uses the last one to allow autoexposure to settle."""
    import os
    
    # Number of frames to capture for autoexposure settling
    num_frames = 3
    # Framerate to request from camera (set to None to use camera's default)
    # Common values: 5, 10, 15, 30 fps (check what your camera supports)
    framerate = 10  # 10 fps as supported by camera
    # Resolution settings
    video_width = 1280
    video_height = 720
    
    print(f"Capturing frame from video device ({num_frames} frames for autoexposure at {video_width}x{video_height} @ {framerate}fps)...")
    
    # Ensure YOLO input directory exists
    YOLO_INPUT.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        last_frame_idx = num_frames - 1  # 0-indexed, so for 3 frames, last is frame 2
        
        # Optimized: Capture frames directly to YOLO input directory
        # This eliminates the need for a temporary file, second ffmpeg call, and copy operation
        # Use list format to avoid shell parsing issues with the filter expression
        cmd_capture = [
            "ffmpeg",
            "-f", "video4linux2",
        ]
        
        # Add framerate option if specified (must come before -i)
        if framerate is not None:
            cmd_capture.extend(["-framerate", str(framerate)])
        
        # Add video size (must come before -i)
        cmd_capture.extend(["-video_size", f"{video_width}x{video_height}"])
        
        # Add input and processing options
        # ffmpeg's filter parser requires comma to be escaped as \, even in list format
        # Chain filters: select the last frame, flip it vertically, then decrease brightness by 15%
        filter_expr = f"select=gte(n\\,{last_frame_idx}),vflip,eq=brightness=-0.15"
        cmd_capture.extend([
            "-i", VIDEO_DEVICE,
            "-vframes", str(num_frames),  # Process num_frames to allow autoexposure
            "-vf", filter_expr,
            "-frames:v", "1",  # Write only one frame to output
            "-vsync", "0",
            "-y",
            str(YOLO_INPUT)  # Write directly to YOLO input directory
        ])
        
        result = subprocess.run(
            cmd_capture,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=5  # Increased timeout to allow for camera initialization
        )
        
        # Verify the output frame file was created
        if not os.path.exists(YOLO_INPUT):
            print(f"Error: Frame file was not created at {YOLO_INPUT}")
            return False
        
        # Check file size to ensure it's not empty
        if os.path.getsize(YOLO_INPUT) == 0:
            print(f"Error: Frame file is empty at {YOLO_INPUT}")
            return False
        
        print(f"Frame captured successfully (using last of {num_frames} frames, frame {last_frame_idx}): {YOLO_INPUT}")
        return True
            
    except subprocess.TimeoutExpired:
        print("Error: Frame capture timed out after 5 seconds")
        return False
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        print(f"Error capturing frame: {error_msg}")
        # Print first 500 chars of stderr for debugging
        if error_msg:
            print(f"FFmpeg error details: {error_msg[:500]}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
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
    
    # Copy result image (use copy instead of copy2 for speed)
    result_image = RESULTS_DIR / f"result_{timestamp}.png"
    try:
        if YOLO_RESULT.exists():
            shutil.copy(YOLO_RESULT, result_image)
        else:
            print(f"Warning: Result image not found at {YOLO_RESULT}")
    except Exception as e:
        print(f"Error copying result image: {e}")
    
    # Save detection labels
    result_labels = RESULTS_DIR / f"result_{timestamp}.txt"
    try:
        with open(result_labels, 'w') as f:
            f.write('\n'.join(detection_lines))
            if detection_lines:  # Add newline at end if there are lines
                f.write('\n')
        return True
    except Exception as e:
        print(f"Error saving detection labels: {e}")
        return False


def main():
    """Main execution function."""
    print("=" * 60)
    print("YOLOv5 Inference Pipeline")
    print("=" * 60)
    
    # Step 1: Capture frame directly to YOLO input directory
    if not capture_frame():
        print("Failed to capture frame. Exiting.")
        return
    
    # Step 2: Run YOLO inference
    detection_lines, success = run_yolo_inference()
    if not success:
        print("Failed to run YOLO inference. Exiting.")
        return
    
    # Step 3: Save results
    save_results(detection_lines)
    
    print("=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

