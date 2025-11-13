import serial
import time
import json
import os
from datetime import datetime

# === CONFIG ===
PORT = "/dev/cu.usbmodem5A450483901"   # change for your OS
BAUD = 115200
OUTPUT_DIR = "dataset"
CAPTURE_TIME = 10  # seconds

os.makedirs(OUTPUT_DIR, exist_ok=True)

def collect_data(label, duration=CAPTURE_TIME):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(OUTPUT_DIR, f"{label}_{timestamp}")
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, "data.jsonl")

    ser = serial.Serial(PORT, BAUD, timeout=2)
    time.sleep(1)
    ser.reset_input_buffer()

    print(f">> Starting {duration}s capture for label '{label}'...")
    ser.write(f"START {duration}\n".encode())

    start = time.time()
    with open(file_path, "w") as f:
        while time.time() - start < duration + 2:
            if ser.in_waiting:
                line = ser.readline().decode(errors="ignore").strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        json.loads(line)
                        f.write(line + "\n")
                    except json.JSONDecodeError:
                        continue
    ser.close()
    print(f"Saved to {file_path}")

if __name__ == "__main__":
    while True:
        label = input("Enter label (e.g. noise, human, animal, etc.) or 'q' to quit: ").strip()
        if label.lower() == "q":
            break
        
        # print countdown
        for i in range(5, 0, -1):
            print(f"Starting in {i}...", end="\r")
            time.sleep(1)
        print("#" * 20, end="\r")  # clear line
        
        collect_data(label)
