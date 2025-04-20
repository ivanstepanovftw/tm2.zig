import math

input_path = "/home/i/Downloads/Telegram Desktop/IMDBTrainingData.txt"
output_path = "IMDBTrainingData.bin"
# input_path = "/home/i/Downloads/Telegram Desktop/IMDBTestData.txt"
# output_path = "IMDBTestData.bin"

with open(input_path, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

# Проверим, что все строки одной длины
feature_counts = [len(line.split()) for line in lines]
if len(set(feature_counts)) != 1:
    raise ValueError("В файле строки имеют разное количество фичей.")

bits_per_sample = feature_counts[0]  # включая таргет
bytes_per_sample = math.ceil(bits_per_sample / 8)

print(f"Битов на пример (включая таргет): {bits_per_sample}")
print(f"Байт на пример (с паддингом): {bytes_per_sample}")

with open(output_path, "wb") as out_f:
    for line in lines:
        bits = ''.join(line.split())
        # Дополняем нулями справа до целого числа байтов
        bits = bits.ljust(bytes_per_sample * 8, '0')
        # Преобразуем каждую восьмерку битов в байт и записываем
        for i in range(0, len(bits), 8):
            byte = bits[i:i+8]
            out_f.write(int(byte, 2).to_bytes(1, byteorder='big'))

print(f"Бинарный файл записан: {output_path}")
