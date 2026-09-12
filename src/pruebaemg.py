import serial
import time
import csv

PORT = "COM4"
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=1)

time.sleep(2)

print("Leyendo datos del EMG...")

try:
    with open("raw1.csv", "w", newline="") as f:
        writer = csv.writer(f)

        while True:
            line = ser.readline().decode(errors="ignore").strip()

            if line:
                print(line)
                writer.writerow([time.time(), line])

except KeyboardInterrupt:
    print("\nGrabación detenida por el usuario.")

finally:
    ser.close()
    print("Puerto serie cerrado correctamente.")
