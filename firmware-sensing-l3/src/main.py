#!/usr/bin/env python3
"""
Main script that monitors GPIO for pulse pattern triggers and runs CNN inference.
Uses rising edge detection to identify two event types:
- Event 1: Single rising edge (no second edge within timeout) - captures frame only
- Event 2: Two rising edges within timeout period - captures frame and runs CNN

Detection logic:
1. Detect first rising edge (LOW -> HIGH transition)
2. Wait for second rising edge within PULSE_TIMEOUT_MS
3. If second edge detected within timeout -> Event 2
4. If timeout reached without second edge -> Event 1
"""

import subprocess
import time
import signal
import sys
from pathlib import Path
from datetime import datetime
import shutil
import wiringpi
import cnn
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# GPIO Configuration (wiringPi numbering)
GPIO_INPUT_PIN = 2  # wiringPi pin 2 (BCM GPIO 27, physical pin 13)
GPIO_OUTPUT_PIN_DETERRENCE = 3  # wiringPi pin 3 (BCM GPIO 22, physical pin 15) - output for deterrence
GPIO_OUTPUT_PIN_LIGHTS = 4  # wiringPi pin 4 (BCM GPIO 23, physical pin 16) - output for lights
PULSE_TIMEOUT_MS = 80  # Timeout for detecting second rising edge (ms)
DEBOUNCE_TIME = 0.02  # Minimum seconds between triggers

# Path to CNN script
SCRIPT_DIR = Path(__file__).parent
CNN_SCRIPT = SCRIPT_DIR / "cnn.py"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
CAPTURE_OUTPUT = SCRIPT_DIR / "output.jpg"

# Global variables
last_trigger_time = 0
running = True

# Pulse detection variables
first_rising_edge_time = None  # Time of first rising edge (ms)
waiting_for_second_edge = False  # Whether we're waiting for second rising edge
last_gpio_state = wiringpi.LOW  # Previous GPIO state for edge detection


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
    
    # Find all result text files (event2 suffix created by cnn.save_results)
    result_files = list(results_dir.glob("*_event2.txt"))
    if not result_files:
        return None
    
    # Return the most recently modified file
    return max(result_files, key=lambda p: p.stat().st_mtime)


def save_event1_frame():
    """Save the captured frame to results directory with event1 prefix and update output.jpg."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Source is the captured frame (YOLO_INPUT)
    source_frame = cnn.YOLO_INPUT
    
    if not source_frame.exists():
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error: Source frame not found at {source_frame}")
        return False
    
    try:
        # Save to results with event1 suffix
        event1_image = RESULTS_DIR / f"{timestamp}_event1.jpg"
        shutil.copy(source_frame, event1_image)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved Event 1 frame: {event1_image.name}")
        
        # Update output.jpg to match
        shutil.copy(source_frame, CAPTURE_OUTPUT)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Updated output.jpg")
        return True
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error saving Event 1 frame: {e}")
        return False


def capture_frame_only():
    """Capture a frame from the camera without running CNN inference."""
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Event 1 detected - Capturing frame only...")
    try:
        # Set lights HIGH before capturing frame
        set_lights_high()
        try:
            if cnn.capture_frame():
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Frame captured successfully")
                # Trigger deterrence GPIO output (always for Event 1)
                trigger_deterrence_gpio(100)
                # Save frame to results with event1 prefix and update output.jpg
                save_event1_frame()
                return True
            else:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Frame capture failed")
                return False
        finally:
            # Always turn lights off after capture attempt
            set_lights_low()
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error capturing frame: {e}")
        set_lights_low()  # Ensure lights are turned off on error
        return False


def trigger_deterrence_gpio(duration_ms=100):
    """Set GPIO_OUTPUT_PIN_DETERRENCE HIGH for specified duration in milliseconds."""
    try:
        wiringpi.digitalWrite(GPIO_OUTPUT_PIN_DETERRENCE, wiringpi.HIGH)
        time.sleep(duration_ms / 1000.0)
        wiringpi.digitalWrite(GPIO_OUTPUT_PIN_DETERRENCE, wiringpi.LOW)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] GPIO deterrence pin {GPIO_OUTPUT_PIN_DETERRENCE} set HIGH for {duration_ms}ms")
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error controlling GPIO deterrence output: {e}")


def set_lights_high():
    """Set GPIO_OUTPUT_PIN_LIGHTS HIGH."""
    try:
        wiringpi.digitalWrite(GPIO_OUTPUT_PIN_LIGHTS, wiringpi.HIGH)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] GPIO lights pin {GPIO_OUTPUT_PIN_LIGHTS} set HIGH")
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error setting GPIO lights HIGH: {e}")


def set_lights_low():
    """Set GPIO_OUTPUT_PIN_LIGHTS LOW."""
    try:
        wiringpi.digitalWrite(GPIO_OUTPUT_PIN_LIGHTS, wiringpi.LOW)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] GPIO lights pin {GPIO_OUTPUT_PIN_LIGHTS} set LOW")
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error setting GPIO lights LOW: {e}")


def save_event2_result():
    """Save the CNN-processed result image to results directory with event2 prefix and update output.jpg."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Source is the CNN result image (YOLO_RESULT)
    source_result = cnn.YOLO_RESULT
    
    if not source_result.exists():
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error: CNN result image not found at {source_result}")
        return False
    
    try:
        # Save to results with event2 suffix (keep as PNG)
        event2_image = RESULTS_DIR / f"{timestamp}_event2.png"
        shutil.copy(source_result, event2_image)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved Event 2 result: {event2_image.name}")
        
        # Update output.jpg - convert PNG to JPG if PIL is available, otherwise just copy
        if PIL_AVAILABLE:
            try:
                img = Image.open(source_result)
                # Convert RGBA to RGB if necessary (PNG might have alpha channel)
                if img.mode == 'RGBA':
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    rgb_img.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                    img = rgb_img
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(CAPTURE_OUTPUT, 'JPEG', quality=95)
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Updated output.jpg (converted from PNG)")
            except Exception as e:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Warning: Could not convert PNG to JPG: {e}, copying PNG instead")
                shutil.copy(source_result, CAPTURE_OUTPUT.with_suffix('.png'))
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Updated output.png")
        else:
            # PIL not available, just copy the PNG
            shutil.copy(source_result, CAPTURE_OUTPUT.with_suffix('.png'))
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Updated output.png (PIL not available for conversion)")
        return True
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error saving Event 2 result: {e}")
        return False


def trigger_cnn():
    """Execute the CNN inference script and check for human/animal detections.
    Returns True if an animal was detected, False otherwise."""
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Event 2 detected - Running CNN inference...")
    try:
        # Set lights HIGH before capturing frame (CNN script will capture frame)
        set_lights_high()
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
            human_detected = any(det["label"] == "person" and det["confidence"] >= 0.55 for det in detections)
            if human_detected:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Human Detected")
            
            # Check for animal detections
            animal_labels = ["teddy bear", "toy", "animal", "squirrel", "groundhog", "raccoon", "cat", "dog"]
            animal_detected = any(det["label"] in animal_labels and det["confidence"] >= 0.30 for det in detections)
            
            # Only trigger deterrence if animal is detected AND no human is detected
            if animal_detected and not human_detected:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Animal Detected (no human) - Triggering deterrence")
                # Trigger deterrence GPIO output pin HIGH for 100ms
                trigger_deterrence_gpio(100)
            elif animal_detected and human_detected:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Animal Detected but Human also present - Deterrence NOT triggered")
            
            # Save CNN result to results with event2 prefix and update output.jpg (after trigger_deterrence_gpio)
            save_event2_result()
            
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] CNN inference completed")
            return animal_detected
        finally:
            # Always turn lights off after CNN processing
            set_lights_low()
    except subprocess.CalledProcessError as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error running CNN script: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        set_lights_low()  # Ensure lights are turned off on error
        return False
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Unexpected error: {e}")
        set_lights_low()  # Ensure lights are turned off on error
        return False


def setup_gpio():
    """Initialize wiringPi and configure the input and output pins."""
    try:
        if wiringpi.wiringPiSetup() != 0:
            print("Error: wiringPiSetup() failed")
            return False
    except Exception as e:
        print(f"Error initializing wiringPi: {e}")
        return False

    # Configure input pin with internal pull-down (idle state is LOW)
    wiringpi.pinMode(GPIO_INPUT_PIN, wiringpi.INPUT)
    wiringpi.pullUpDnControl(GPIO_INPUT_PIN, wiringpi.PUD_DOWN)

    # Configure deterrence output pin
    wiringpi.pinMode(GPIO_OUTPUT_PIN_DETERRENCE, wiringpi.OUTPUT)
    wiringpi.digitalWrite(GPIO_OUTPUT_PIN_DETERRENCE, wiringpi.LOW)

    # Configure lights output pin
    wiringpi.pinMode(GPIO_OUTPUT_PIN_LIGHTS, wiringpi.OUTPUT)
    wiringpi.digitalWrite(GPIO_OUTPUT_PIN_LIGHTS, wiringpi.LOW)

    print(f"✓ wiringPi initialized.")
    print(f"  Input pin: {GPIO_INPUT_PIN} (pull-down enabled, idle=LOW)")
    print(f"  Deterrence output pin: {GPIO_OUTPUT_PIN_DETERRENCE} (initialized to LOW)")
    print(f"  Lights output pin: {GPIO_OUTPUT_PIN_LIGHTS} (initialized to LOW)")
    return True


def detect_pulse_pattern():
    """Detect pulse patterns on GPIO_INPUT_PIN using rising edge detection.
    Returns:
        'event1' if single rising edge detected (no second edge within timeout)
        'event2' if two rising edges detected within timeout
        None if no event detected yet
    """
    global first_rising_edge_time, waiting_for_second_edge, last_gpio_state
    
    current_time_ms = time.time() * 1000.0
    current_state = wiringpi.digitalRead(GPIO_INPUT_PIN)
    
    # Detect rising edge (LOW -> HIGH transition) BEFORE updating last state
    rising_edge = (last_gpio_state == wiringpi.LOW and current_state == wiringpi.HIGH)
    
    # Check for timeout first - if we've been waiting too long, this is Event 1
    if waiting_for_second_edge and first_rising_edge_time is not None:
        time_since_first = current_time_ms - first_rising_edge_time
        if time_since_first > PULSE_TIMEOUT_MS:
            # Timeout reached - this was Event 1 (single pulse)
            waiting_for_second_edge = False
            first_rising_edge_time = None
            # Update state after handling timeout
            last_gpio_state = current_state
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Timeout reached ({time_since_first:.1f}ms) - Event 1!")
            return 'event1'
    
    # Handle rising edges
    if rising_edge:
        if not waiting_for_second_edge:
            # First rising edge detected
            first_rising_edge_time = current_time_ms
            waiting_for_second_edge = True
            last_gpio_state = current_state
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] First rising edge detected, waiting for second edge...")
            return None
        else:
            # Second rising edge detected - check if within timeout window
            time_since_first = current_time_ms - first_rising_edge_time
            if time_since_first <= PULSE_TIMEOUT_MS:
                # Event 2 detected: two rising edges within timeout
                waiting_for_second_edge = False
                first_rising_edge_time = None
                last_gpio_state = current_state
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Second rising edge detected ({time_since_first:.1f}ms after first) - Event 2!")
                return 'event2'
            else:
                # Edge case: second edge arrived but timeout was exceeded
                # This can happen if the edge arrives right at the boundary
                # Treat as Event 1 (timeout) and start new detection with this edge
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Second edge after timeout ({time_since_first:.1f}ms) - Event 1, starting new detection")
                first_rising_edge_time = current_time_ms
                # Keep waiting_for_second_edge = True for new detection
                last_gpio_state = current_state
                return 'event1'
    
    # Update last state if no edge detected and no timeout
    last_gpio_state = current_state
    return None


def main():
    """Main execution function."""
    global running, last_trigger_time, waiting_for_second_edge, first_rising_edge_time, last_gpio_state
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("=" * 60)
    print("GPIO Monitor for CNN Inference")
    print("=" * 60)
    print(f"Input Pin: {GPIO_INPUT_PIN} (wiringPi numbering)")
    print(f"Deterrence Output Pin: {GPIO_OUTPUT_PIN_DETERRENCE} (wiringPi numbering)")
    print(f"Lights Output Pin: {GPIO_OUTPUT_PIN_LIGHTS} (wiringPi numbering)")
    print(f"CNN Script: {CNN_SCRIPT}")
    print(f"Debounce Time: {DEBOUNCE_TIME} seconds")
    print(f"Pulse Timeout: {PULSE_TIMEOUT_MS}ms")
    print("=" * 60)
    print("Event 1: Single rising edge (no second edge within timeout) -> Capture frame only")
    print("Event 2: Two rising edges within timeout -> Capture + CNN inference")
    print("=" * 60)
    print("Note: GPIO access may require root privileges (sudo)")
    print("=" * 60)
    
    if not setup_gpio():
        print("Failed to setup GPIO. Exiting.")
        sys.exit(1)
    
    # Initialize GPIO state
    last_gpio_state = wiringpi.digitalRead(GPIO_INPUT_PIN)
    
    print("Monitoring GPIO for rising edge patterns (idle state: LOW)...")
    print("Press Ctrl+C to exit\n")
    
    try:
        while running:
            event = detect_pulse_pattern()
            
            if event:
                current_time = time.time()
                # Debouncing: ignore events if too soon after last trigger
                if current_time - last_trigger_time >= DEBOUNCE_TIME:
                    last_trigger_time = current_time
                    
                    if event == 'event1':
                        capture_frame_only()
                    elif event == 'event2':
                        trigger_cnn()
                else:
                    remaining_time = DEBOUNCE_TIME - (current_time - last_trigger_time)
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Event ignored (debounce: {remaining_time:.2f}s remaining)")
                    # Reset detection state after ignoring
                    waiting_for_second_edge = False
                    first_rising_edge_time = None
            
            time.sleep(0.001)  # 1ms polling interval for precise edge detection
    finally:
        # Ensure output pins are set to LOW on exit
        try:
            wiringpi.digitalWrite(GPIO_OUTPUT_PIN_DETERRENCE, wiringpi.LOW)
            wiringpi.digitalWrite(GPIO_OUTPUT_PIN_LIGHTS, wiringpi.LOW)
        except:
            pass
        print("GPIO monitor exiting")


if __name__ == "__main__":
    main()
