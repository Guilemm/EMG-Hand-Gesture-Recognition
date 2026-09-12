import serial
import time
import csv

PORT = "COM4"
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=1)

time.sleep(2)

print("Reading EMG data...")

try:
    with open("raw1.csv", "w", newline="") as f:
        writer = csv.writer(f)

        while True:
            line = ser.readline().decode(errors="ignore").strip()

            if line:
                print(line)
                writer.writerow([time.time(), line])

except KeyboardInterrupt:
    print("\nRecording stopped by user.")

finally:
    ser.close()
    print("Serial port closed successfully.")
