import serial

ser = serial.Serial("/dev/cu.usbserial-1140", 9600)

while True:
    try:
        data = ser.readline().decode("utf-8").strip()  # Читаем строку, декодируем и убираем пробелы
        print(data)  # Выводим в консоль
    except KeyboardInterrupt:
        print("\nПрограмма завершена.")
        break
    except Exception as e:
        print(f"Ошибка: {e}")

ser.close()