import serial
import time
import csv

PORT = "COM4"   # cambia si tu Arduino está en otro puerto, creo que si es el COM4, pero podria no serlo
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=1)

time.sleep(2)   # espera a que Arduino reinicie

print("Leyendo datos del EMG...")

try:
    with open("raw1.csv", "w", newline="") as f: #Cambiar nombre del archivo para crear uno nuevo desde cero, para terminar la grabacion hacer control + C en la terminal, pongo la "w", para que se sobreescriba
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
#Se supone que las medidas que tome se guardaran en la misma carpeta en la que esta este prueba_emg.py        