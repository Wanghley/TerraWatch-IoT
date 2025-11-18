import serial
import time
import json
import sys
from datetime import datetime

# --- Configuration ---
SERIAL_PORT = "COM6"  # <-- CHANGE THIS to your ESP32's port
BAUD_RATE = 115200
# ---------------------

def get_serial_connection(port, baud):
    """Tries to connect to the serial port."""
    try:
        ser = serial.Serial(port, baud, timeout=1.0)
        print(f"Connected to {port} at {baud} baud.")
        time.sleep(2) # Wait for ESP32 to stabilize
        ser.flushInput()
        return ser
    except serial.SerialException:
        print(f"Error: Could not open port {port}.")
        print("Please check the port name and ensure the device is not in use.")
        sys.exit(1)

def collect_data(ser, label, duration_sec, ignore_frames=3):
    """Collects data for a specified duration and saves it to a file."""
    
    # Create a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data_label_{label}_{timestamp}.jsonl"
    print(f"Collecting data for {duration_sec} seconds. Label: {label}")
    print(f"Saving to: {filename}")

    start_time = time.time()
    end_time = start_time + duration_sec
    frame_count = 0
    ignored = 0

    try:
        with open(filename, 'w') as f:
            ser.flushInput() # Clear old data
            while time.time() < end_time:
                try:
                    line = ser.readline().decode('utf-8').strip()
                    if not line:
                        continue

                    if ignored < ignore_frames:
                        ignored += 1
                        continue

                    # Try to parse the JSON
                    if line.startswith('{') and line.endswith('}'):
                        data = json.loads(line)
                        # Add a timestamp to the data
                        data['capture_timestamp_ms'] = int(time.time() * 1000)
                        
                        # Write the JSON object as a single line
                        f.write(json.dumps(data) + '\n')
                        frame_count += 1
                    
                except json.JSONDecodeError:
                    print(f"Warning: Corrupt data frame skipped: {line[:50]}...")
                except UnicodeDecodeError:
                    print("Warning: Unicode decode error skipped.")
            
    except IOError as e:
        print(f"Error writing to file: {e}")
    except KeyboardInterrupt:
        print("\nCollection stopped early by user.")
        
    print(f"Collection complete. Saved {frame_count} frames to {filename}.")
    print(f"Actual collection time: {time.time() - start_time:.2f} seconds.")

def main():
    """Main loop for the data collector."""
    
    # Install pyserial if not present
    try:
        import serial
    except ImportError:
        print("Error: 'pyserial' library not found.")
        print("Please install it by running: pip install pyserial")
        sys.exit(1)

    ser = get_serial_connection(SERIAL_PORT, BAUD_RATE)
    
    while True:
        print("\n--- TerraWatch Data Collector ---")
        print(" (1) Collect data for label '1' (e.g., Animal)")
        print(" (0) Collect data for label '0' (e.g., Human/Noise/Other)")
        print(" (q) Quit")
        
        choice = input("Enter your choice: ").strip().lower()
        
        if choice == 'q':
            print("Exiting.")
            ser.close()
            break
            
        if choice in ['0', '1']:
            label = int(choice)
            try:
                duration = int(input(f"Enter duration in seconds for label '{label}': ").strip())
                if duration <= 0:
                    raise ValueError
            except ValueError:
                print("Invalid duration. Please enter a positive number.")
                continue
            
            # Start collection
            collect_data(ser, label, duration)
        else:
            print("Invalid choice. Please enter '0', '1', or 'q'.")

if __name__ == "__main__":
    main()