import csv

# Данные для записи
header = ['product_name', 'price']
rows = [
    ['iphone 16 256gb', 1100],
    ['macbook pro m4 16 inch 2tb',4300],
    ['samsung galaxy s24 ultra 256gb',420],
    ['msi katana 3050 RTX 16 GB RAM', 600],
    ['xiaomi 15 ultra 512 GB', 1200],
    ['Mazda Mazda3 i Sport 4dr Sedan 4A', 1500],
    ['2006 BMW 3-Series I AUTOMATIC', 1500]
]

# Имя файла
filename = 'simple_example.csv'

# Запись данных в CSV-файл
with open(filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(header)    # Запись заголовка
    writer.writerows(rows)     # Запись строк данных

print(f'Файл {filename} успешно создан.')