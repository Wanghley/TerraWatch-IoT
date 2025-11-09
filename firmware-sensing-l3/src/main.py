#!/usr/bin/env python3
"""
Main script that monitors GPIO for rising edge triggers and runs CNN inference.
Includes debouncing to prevent multiple triggers within 3 seconds.
"""

import subprocess
import time
import signal
import sys
from pathlib import Path
import wiringpi

# GPIO Configuration (wiringPi numbering)
GPIO_PIN = 2  # wiringPi pin 2 (BCM GPIO 27, physical pin 13)
DEBOUNCE_TIME = 0.3  # Minimum seconds between triggers

# Path to CNN script
SCRIPT_DIR = Path(__file__).parent
CNN_SCRIPT = SCRIPT_DIR / "cnn.py"

# Global variables
last_trigger_time = 0
running = True


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    global running
    print("\nShutting down GPIO monitor...")
    running = False
    sys.exit(0)


def parse_detection_labels(detection_lines):
    """Parse detection labels from detection lines.
    Detection format: "16:  91%, [ 135,  218,  305,  553], dog"
    Returns a list of detection dictionaries with keys: 'label' (lowercase) and 'confidence' (0-1 float).
    """
    detections = []

    for line in detection_lines:
        line = line.strip()
        # Check if line matches the detection format: "16:  91%, [ 135,  218,  305,  553], dog"
        if ':' in line and '%' in line and '[' in line and ']' in line:
            # Extract confidence (between ":" and "%")
            try:
                before_percent = line.split('%', 1)[0]
                confidence_str = before_percent.split(':', 1)[-1].strip()
                confidence = float(confidence_str) / 100.0
            except (ValueError, IndexError):
                continue

            # Extract label - it's after the last comma
            parts = line.split(',')
            if len(parts) >= 2:
                label = parts[-1].strip().lower()
                if label:
                    detections.append({
                        "label": label,
                        "confidence": confidence
                    })

    return detections


def get_latest_results_file():
    """Get the most recent results file from the results directory."""
    results_dir = SCRIPT_DIR.parent / "results"
    if not results_dir.exists():
        return None
    
    # Find all result text files
    result_files = list(results_dir.glob("result_*.txt"))
    if not result_files:
        return None
    
    # Return the most recently modified file
    return max(result_files, key=lambda p: p.stat().st_mtime)


def trigger_cnn():
    """Execute the CNN inference script and check for human/animal detections."""
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] GPIO trigger detected - Running CNN inference...")
    try:
        result = subprocess.run(
            [sys.executable, str(CNN_SCRIPT)],
            cwd=str(SCRIPT_DIR),
            check=True,
            capture_output=True,
            text=True
        )
        
        # Wait a moment for file to be written, then read the latest results file
        time.sleep(0.1)
        results_file = get_latest_results_file()
        
        detections = []
        if results_file and results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    detection_lines = f.readlines()
                detections = parse_detection_labels(detection_lines)
            except Exception as e:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Warning: Could not read results file: {e}")
        
        # Check for human detection
        if any(det["label"] == "person" and det["confidence"] >= 0.55 for det in detections):
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Human Detected")
        
        # Check for animal detections
        animal_labels = ["teddy bear", "toy", "animal", "squirrel", "groundhog", "raccoon", "cat", "dog"]
        if any(det["label"] in animal_labels and det["confidence"] >= 0.30 for det in detections):
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Animal Detected")
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] CNN inference completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error running CNN script: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Unexpected error: {e}")
        return False


def setup_gpio():
    """Initialize wiringPi and configure the button input."""
    try:
        if wiringpi.wiringPiSetup() != 0:
            print("Error: wiringPiSetup() failed")
            return False
    except Exception as e:
        print(f"Error initializing wiringPi: {e}")
        return False

    # Configure pin as input with internal pull-up so the button can pull it to ground
    wiringpi.pinMode(GPIO_PIN, wiringpi.INPUT)
    wiringpi.pullUpDnControl(GPIO_PIN, wiringpi.PUD_UP)

    print(f"✓ wiringPi initialized. Monitoring wiringPi pin {GPIO_PIN} (pull-up enabled)")
    return True


def main():
    """Main execution function."""
    global running, last_trigger_time
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("=" * 60)
    print("GPIO Monitor for CNN Inference")
    print("=" * 60)
    print(f"GPIO Pin: {GPIO_PIN} (wiringPi numbering)")
    print(f"CNN Script: {CNN_SCRIPT}")
    print(f"Debounce Time: {DEBOUNCE_TIME} seconds")
    print("=" * 60)
    print("Note: GPIO access may require root privileges (sudo)")
    print("=" * 60)
    
    if not setup_gpio():
        print("Failed to setup GPIO. Exiting.")
        sys.exit(1)
    
    print("Monitoring GPIO for button presses (falling edge, pull-up enabled)...")
    print("Press Ctrl+C to exit\n")
    
    last_state = wiringpi.digitalRead(GPIO_PIN)
    
    try:
        while running:
            state = wiringpi.digitalRead(GPIO_PIN)
            if last_state == wiringpi.HIGH and state == wiringpi.LOW:
                current_time = time.time()
                if current_time - last_trigger_time >= DEBOUNCE_TIME:
                    last_trigger_time = current_time
                    trigger_cnn()
                else:
                    remaining_time = DEBOUNCE_TIME - (current_time - last_trigger_time)
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Button press ignored (debounce: {remaining_time:.2f}s remaining)")
            last_state = state
            time.sleep(0.01)
    finally:
        print("GPIO monitor exiting")


if __name__ == "__main__":
    main()
