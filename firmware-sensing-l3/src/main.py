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

# GPIO Configuration
# Change this to the GPIO pin number you want to monitor
# For Orange Pi, use BOARD mode (physical pin number) or BCM mode (GPIO number)
# For Orange Pi 4A, you may need to check the pin mapping
GPIO_PIN = 18  # Change this to your desired GPIO pin
GPIO_MODE = "BOARD"  # "BOARD" for physical pin numbers, "BCM" for GPIO numbers
# Orange Pi board type for OPi.GPIO (if using OPi.GPIO library)
# Common values: "PC2", "ZERO", "ONE", "PLUS2E", "PC", "4" (for Orange Pi 4)
# Set to None to auto-detect or skip board setting
ORANGE_PI_BOARD = None  # None = auto-detect, or set to specific board constant
DEBOUNCE_TIME = 3.0  # Minimum seconds between triggers

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


def trigger_cnn():
    """Execute the CNN inference script."""
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] GPIO trigger detected - Running CNN inference...")
    try:
        result = subprocess.run(
            [sys.executable, str(CNN_SCRIPT)],
            cwd=str(SCRIPT_DIR),
            check=True
        )
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] CNN inference completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error running CNN script: {e}")
        return False
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Unexpected error: {e}")
        return False


def gpio_callback(channel):
    """Callback function for GPIO rising edge detection."""
    global last_trigger_time
    
    current_time = time.time()
    time_since_last_trigger = current_time - last_trigger_time
    
    # Debouncing: only trigger if enough time has passed
    if time_since_last_trigger >= DEBOUNCE_TIME:
        last_trigger_time = current_time
        trigger_cnn()
    else:
        remaining_time = DEBOUNCE_TIME - time_since_last_trigger
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] GPIO trigger ignored (debounce: {remaining_time:.2f}s remaining)")


def setup_gpio():
    """Setup GPIO pin for rising edge detection.
    Optimized for OPi.GPIO library (Orange Pi specific).
    Falls back to other libraries if OPi.GPIO is not available.
    """
    # Try OPi.GPIO first (Orange Pi specific library)
    try:
        import OPi.GPIO as GPIO
        
        # Set board type - for Orange Pi 4A, you may need to adjust this
        # Common options: GPIO.PC2, GPIO.ZERO, GPIO.ONE, GPIO.PLUS2E, GPIO.PC, GPIO.ORANGEPI4
        board_set = False
        if ORANGE_PI_BOARD:
            # Try user-specified board type
            try:
                board_attr = getattr(GPIO, ORANGE_PI_BOARD.upper())
                GPIO.setboard(board_attr)
                print(f"Board type set to: {ORANGE_PI_BOARD}")
                board_set = True
            except (AttributeError, Exception) as e:
                print(f"Warning: Could not set board type to {ORANGE_PI_BOARD}: {e}")
                print("Available board types can be checked in OPi.GPIO documentation")
        
        if not board_set:
            # Try common board types for Orange Pi 4A and other models
            # Order matters - try most likely first
            board_types = ['ORANGEPI4', 'ORANGEPI_4', 'PC2', 'ZERO', 'ONE', 'PLUS2E', 'PC', 'LITE', 'PLUS']
            for board_name in board_types:
                try:
                    board_attr = getattr(GPIO, board_name)
                    GPIO.setboard(board_attr)
                    print(f"Board type auto-detected/set to: {board_name}")
                    board_set = True
                    break
                except (AttributeError, Exception):
                    continue
            
            if not board_set:
                # Some versions of OPi.GPIO don't require setboard, or use different method
                try:
                    # Try to proceed without setting board - some versions work this way
                    print("Note: Board type not set, using library default")
                    print("If you encounter issues, try setting ORANGE_PI_BOARD in the script")
                except:
                    pass
        
        # Set mode based on configuration
        try:
            if GPIO_MODE.upper() == "BOARD":
                GPIO.setmode(GPIO.BOARD)
                mode_str = "BOARD (physical pin)"
            else:
                GPIO.setmode(GPIO.BCM)
                mode_str = "BCM (GPIO number)"
        except Exception as e:
            print(f"Error setting GPIO mode: {e}")
            raise
        
        # Setup pin as input with pull-down resistor
        try:
            GPIO.setup(GPIO_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        except Exception as e:
            print(f"Error setting up GPIO pin {GPIO_PIN}: {e}")
            print("Make sure the pin number is correct for your board type and mode")
            raise
        
        # Add event detection for rising edge with hardware debounce
        try:
            GPIO.add_event_detect(
                GPIO_PIN,
                GPIO.RISING,
                callback=gpio_callback,
                bouncetime=100  # Hardware debounce in milliseconds (prevents noise)
            )
        except Exception as e:
            print(f"Error setting up event detection: {e}")
            raise
        
        print(f"✓ GPIO {GPIO_PIN} configured using OPi.GPIO for rising edge detection ({mode_str})")
        return True, ("OPi", GPIO)
        
    except ImportError:
        # Try OrangePi.GPIO (alternative Orange Pi library)
        try:
            import OrangePi.GPIO as GPIO
            
            # Set mode
            if GPIO_MODE.upper() == "BOARD":
                GPIO.setmode(GPIO.BOARD)
                mode_str = "BOARD (physical pin)"
            else:
                GPIO.setmode(GPIO.BCM)
                mode_str = "BCM (GPIO number)"
            
            GPIO.setup(GPIO_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
            
            GPIO.add_event_detect(
                GPIO_PIN,
                GPIO.RISING,
                callback=gpio_callback,
                bouncetime=100
            )
            
            print(f"GPIO {GPIO_PIN} configured using OrangePi.GPIO for rising edge detection ({mode_str})")
            return True, ("OrangePi", GPIO)
            
        except ImportError:
            # Try gpiod (modern Linux GPIO character device - works on all Linux systems)
            try:
                import gpiod
                
                # For gpiod, we need the GPIO chip and line offset
                # On Orange Pi, typically gpiochip0, but may vary
                chip = gpiod.Chip('gpiochip0')
                line = chip.get_line(GPIO_PIN)
                line.request(consumer='cnn_trigger', type=gpiod.LINE_REQ_EV_RISING_EDGE)
                
                print(f"GPIO {GPIO_PIN} configured using gpiod for rising edge detection")
                print("Note: gpiod uses GPIO chip line numbers, not physical pins")
                return True, ("gpiod", (chip, line))
                
            except ImportError:
                # Last resort: Try RPi.GPIO (unlikely to work on Orange Pi, but worth trying)
                try:
                    import RPi.GPIO as GPIO
                    
                    if GPIO_MODE.upper() == "BOARD":
                        GPIO.setmode(GPIO.BOARD)
                        mode_str = "BOARD (physical pin)"
                    else:
                        GPIO.setmode(GPIO.BCM)
                        mode_str = "BCM (GPIO number)"
                    
                    GPIO.setup(GPIO_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
                    
                    GPIO.add_event_detect(
                        GPIO_PIN,
                        GPIO.RISING,
                        callback=gpio_callback,
                        bouncetime=100
                    )
                    
                    print(f"GPIO {GPIO_PIN} configured using RPi.GPIO for rising edge detection ({mode_str})")
                    print("Warning: RPi.GPIO may not work correctly on Orange Pi")
                    return True, ("RPi", GPIO)
                    
                except ImportError:
                    print("Error: No GPIO library found.")
                    print("Please install one of the following:")
                    print("  - OPi.GPIO (recommended for Orange Pi): pip install OPi.GPIO")
                    print("  - OrangePi.GPIO: pip install OrangePi.GPIO")
                    print("  - gpiod (universal Linux): sudo apt-get install libgpiod-dev && pip install gpiod")
                    print("  - RPi.GPIO: pip install RPi.GPIO (may not work on Orange Pi)")
                    return False, None
            except Exception as e:
                print(f"Error setting up gpiod: {e}")
                print("You may need to run with sudo or adjust GPIO chip name")
                return False, None
        except Exception as e:
            print(f"Error setting up OrangePi.GPIO: {e}")
            return False, None
    except Exception as e:
        print(f"Error setting up OPi.GPIO: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure OPi.GPIO is installed: pip install OPi.GPIO")
        print("2. Try running with sudo: sudo python3 main.py")
        print("3. Check if the GPIO pin number is correct for your board")
        print("4. Try setting ORANGE_PI_BOARD to a specific board type")
        print("5. Verify your Orange Pi model and check OPi.GPIO documentation")
        return False, None


def monitor_gpio_gpiod(chip, line):
    """Monitor GPIO using gpiod library."""
    import gpiod  # Import here since it's only needed when using gpiod
    global running, last_trigger_time
    
    print("Monitoring GPIO for rising edge triggers...")
    print(f"Debounce time: {DEBOUNCE_TIME} seconds")
    print("Press Ctrl+C to exit\n")
    
    while running:
        if line.event_wait(timeout=1.0):
            event = line.event_read()
            if event.type == gpiod.LineEvent.RISING_EDGE:
                current_time = time.time()
                time_since_last_trigger = current_time - last_trigger_time
                
                if time_since_last_trigger >= DEBOUNCE_TIME:
                    last_trigger_time = current_time
                    trigger_cnn()
                else:
                    remaining_time = DEBOUNCE_TIME - time_since_last_trigger
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] GPIO trigger ignored (debounce: {remaining_time:.2f}s remaining)")


def main():
    """Main execution function."""
    global running
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("=" * 60)
    print("GPIO Monitor for CNN Inference")
    print("=" * 60)
    print(f"GPIO Pin: {GPIO_PIN} ({GPIO_MODE} mode)")
    print(f"CNN Script: {CNN_SCRIPT}")
    print(f"Debounce Time: {DEBOUNCE_TIME} seconds")
    if ORANGE_PI_BOARD:
        print(f"Board Type: {ORANGE_PI_BOARD} (manual)")
    else:
        print(f"Board Type: Auto-detect")
    print("=" * 60)
    print("Note: GPIO access may require root privileges (sudo)")
    print("=" * 60)
    
    # Setup GPIO
    success, gpio_obj = setup_gpio()
    if not success:
        print("Failed to setup GPIO. Exiting.")
        sys.exit(1)
    
    # Check which GPIO library is being used
    gpio_type, gpio_handle = gpio_obj
    
    if gpio_type == "gpiod":
        # Using gpiod - need to poll for events
        chip, line = gpio_handle
        try:
            monitor_gpio_gpiod(chip, line)
        finally:
            line.release()
            chip.close()
            print("GPIO cleaned up")
    else:
        # Using OPi.GPIO, OrangePi.GPIO, or RPi.GPIO - callback-based, just wait
        GPIO = gpio_handle
        print("Monitoring GPIO for rising edge triggers...")
        print(f"Debounce time: {DEBOUNCE_TIME} seconds")
        print("Press Ctrl+C to exit\n")
        
        try:
            while running:
                time.sleep(0.1)
        finally:
            GPIO.cleanup()
            print("GPIO cleaned up")


if __name__ == "__main__":
    main()
