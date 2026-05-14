import sys
import csv

def read_csv(filename):
    """Чтение CSV файла в список списков"""
    matrix = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            matrix.append([float(x) for x in row if x.strip()])
    return matrix

def max_difference(file1, file2):
    """Вычисление максимальной разности между двумя CSV файлами"""
    m1 = read_csv(file1)
    m2 = read_csv(file2)
    
    # Проверяем размеры
    if len(m1) != len(m2) or len(m1[0]) != len(m2[0]):
        print(f"Предупреждение: размеры не совпадают!", file=sys.stderr)
        print(f"  Файл1: {len(m1)}x{len(m1[0])}", file=sys.stderr)
        print(f"  Файл2: {len(m2)}x{len(m2[0])}", file=sys.stderr)
    
    # Сравниваем по минимальным размерам
    rows = min(len(m1), len(m2))
    cols = min(len(m1[0]), len(m2[0]))
    
    max_diff = 0
    max_pos = (0, 0)
    
    for i in range(rows):
        for j in range(cols):
            diff = abs(m1[i][j] - m2[i][j])
            if diff > max_diff:
                max_diff = diff
                max_pos = (i, j)
    
    return max_diff, max_pos, m1[max_pos[0]][max_pos[1]], m2[max_pos[0]][max_pos[1]]

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare.py file1.csv file2.csv")
        sys.exit(1)
    
    diff, pos, v1, v2 = max_difference(sys.argv[1], sys.argv[2])
    print(f"Max difference: {diff}")
    print(f"At position: {pos}")
    print(f"Values: {v1} vs {v2}")