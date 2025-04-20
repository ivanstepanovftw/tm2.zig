import math

# input_path = "IMDBTrainingData.bin"
# output_path = "IMDBTrainingData_decoded.txt"
input_path = "/home/i/Downloads/Telegram Desktop/IMDBTrainingData.bin"
output_path = "IMDBTrainingData_zig.txt"

# Укажи здесь количество битов на пример (включая таргет)
# Битов на пример (включая таргет): 40001
# Байт на пример (с паддингом): 5001
bits_per_sample = 40001  # <--- ПОМЕНЯЙ на актуальное значение
bytes_per_sample = math.ceil(bits_per_sample / 8)

with open(input_path, "rb") as f:
    binary_data = f.read()

if len(binary_data) % bytes_per_sample != 0:
    raise ValueError("Длина бинарного файла не кратна размеру одного примера!")

num_samples = len(binary_data) // bytes_per_sample

with open(output_path, "w") as out_f:
    for i in range(num_samples):
        chunk = binary_data[i * bytes_per_sample:(i + 1) * bytes_per_sample]
        # Преобразуем байты в битовую строку
        bits = ''.join(f"{byte:08b}" for byte in chunk)
        # Отрезаем паддинг
        bits = bits[:bits_per_sample]
        # Добавим пробелы между битами
        line = ' '.join(bits)
        out_f.write(line + '\n')

print(f"Декодированный .txt файл сохранён: {output_path}")
