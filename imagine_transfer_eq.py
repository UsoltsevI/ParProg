"""
Тепловая визуализация данных уравнения теплопроводности
Читает CSV файл с матрицей температур (строки = время, столбцы = пространство)
Создает цветную тепловую карту с автоматической нормализацией
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import sys
import os

# ==================== ЗАГРУЗКА ДАННЫХ ====================

def load_heat_data(filename):
    """
    Загружает данные из CSV файла, созданного C программой
    
    Формат: строки = время, столбцы = пространство
    Поддерживает комментарии (#) и пустые строки
    
    Parameters:
    -----------
    filename : str
        Путь к CSV файлу
        
    Returns:
    --------
    numpy.ndarray : 2D массив температур [время, пространство]
    """
    try:
        # Пытаемся прочитать как обычный CSV
        data = np.loadtxt(filename, delimiter=',', comments='#')
        
        # Если данные в одной строке, возможно нужна транспонирование
        if data.ndim == 1:
            data = data.reshape(1, -1)
            
        print(f"✓ Загружены данные: {data.shape[0]} временных слоев × {data.shape[1]} пространственных точек")
        print(f"  Диапазон температур: [{np.min(data):.4f}, {np.max(data):.4f}]")
        
        return data
        
    except Exception as e:
        print(f"✗ Ошибка загрузки файла: {e}")
        print("  Проверьте формат файла. Должен быть CSV с числами, разделенными запятыми")
        sys.exit(1)

# ==================== ЦВЕТОВЫЕ СХЕМЫ ====================

def get_custom_colormap(name='thermal'):
    """
    Возвращает кастомную цветовую карту для тепловизора
    
    Parameters:
    -----------
    name : str
        Название цветовой схемы: 'thermal', 'jet', 'coolwarm', 'plasma'
        
    Returns:
    --------
    matplotlib.colors.Colormap
    """
    if name == 'thermal':
        # Цвета тепловизора (синий -> голубой -> зеленый -> желтый -> красный)
        colors_list = ['#00008B', '#0000FF', '#00FFFF', '#00FF00', 
                       '#FFFF00', '#FFA500', '#FF0000', '#8B0000']
        return LinearSegmentedColormap.from_list('thermal', colors_list, N=256)
    
    elif name == 'jet':
        # Стандартная jet colormap (как в MATLAB)
        colors_list = ['#00008B', '#0000FF', '#00FFFF', '#00FF00', 
                       '#FFFF00', '#FFA500', '#FF0000']
        return LinearSegmentedColormap.from_list('jet', colors_list, N=256)
    
    elif name == 'coolwarm':
        # Синий-белый-красный (хорошо для симметричных данных)
        return plt.cm.coolwarm
    
    elif name == 'plasma':
        # Современная цветовая схема
        return plt.cm.plasma
    
    else:
        # По умолчанию hot
        return plt.cm.hot

# ==================== ВИЗУАЛИЗАЦИЯ ====================

def create_heatmap(temperature, title="Распределение температуры", 
                   cmap_name='thermal', save_path=None, dpi=150):
    """
    Создает основную тепловую карту с автоматической нормализацией
    
    Parameters:
    -----------
    temperature : numpy.ndarray
        2D массив температур
    title : str
        Заголовок графика
    cmap_name : str
        Название цветовой схемы ('thermal', 'jet', 'coolwarm', 'plasma')
    save_path : str or None
        Путь для сохранения (если None, только показывает)
    dpi : int
        Разрешение для сохранения
    """
    # Нормализуем значения
    vmin, vmax = np.min(temperature), np.max(temperature)
    
    # Создаем фигуру
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Отрисовываем тепловую карту с нормализацией
    cmap = get_custom_colormap(cmap_name)
    im = ax.imshow(temperature, aspect='auto', origin='lower', 
                   cmap=cmap, interpolation='bilinear',
                   vmin=vmin, vmax=vmax)
    
    # Добавляем цветовую шкалу
    cbar = plt.colorbar(im, ax=ax, label='Температура', 
                        orientation='vertical', pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    
    # Настройка осей
    ax.set_xlabel('Пространственная координата x', fontsize=12, fontweight='bold')
    ax.set_ylabel('Временная координата t', fontsize=12, fontweight='bold')
    ax.set_title(f'{title}\n(нормализовано: [{vmin:.3f}, {vmax:.3f}])', 
                 fontsize=14, pad=20)
    
    # Добавляем сетку (опционально)
    ax.grid(False)
    
    # Добавляем информацию о данных
    textstr = f'Min: {vmin:.4f}\nMax: {vmax:.4f}\nMean: {np.mean(temperature):.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"✓ Рисунок сохранен: {save_path}")
    
    plt.show()
    
    return fig, ax

def create_enhanced_visualization(temperature, cmap_name='thermal', save_path=None):
    """
    Создает расширенную визуализацию:
    - Основная тепловая карта
    - Временные профили в разных точках
    - Гистограмма распределения температур
    """
    vmin, vmax = np.min(temperature), np.max(temperature)
    time_steps, space_points = temperature.shape
    
    # Создаем фигуру с подграфиками
    fig = plt.figure(figsize=(16, 10))
    
    # Создаем сетку: 2x2
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.6], width_ratios=[1, 0.8])
    
    # 1. Основная тепловая карта
    ax1 = fig.add_subplot(gs[0, 0])
    cmap = get_custom_colormap(cmap_name)
    im = ax1.imshow(temperature, aspect='auto', origin='lower', 
                    cmap=cmap, interpolation='bilinear',
                    vmin=vmin, vmax=vmax)
    ax1.set_xlabel('Пространство x', fontsize=11)
    ax1.set_ylabel('Время t', fontsize=11)
    ax1.set_title('Тепловая карта', fontsize=12, fontweight='bold')
    
    # Цветовая шкаба для тепловой карты
    cbar1 = plt.colorbar(im, ax=ax1, label='Температура', fraction=0.046, pad=0.04)
    
    # 2. Эволюция температуры во времени в разных точках пространства
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Выбираем несколько пространственных точек
    n_points = min(8, space_points)
    points_idx = np.linspace(0, space_points-1, n_points, dtype=int)
    
    for idx in points_idx:
        ax2.plot(temperature[:, idx], linewidth=1.5, 
                label=f'x = {idx}')
    
    ax2.set_xlabel('Временной шаг', fontsize=11)
    ax2.set_ylabel('Температура', fontsize=11)
    ax2.set_title('Эволюция температуры\nв разных точках пространства', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # 3. Пространственное распределение в разные моменты времени
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Выбираем несколько временных слоев
    n_times = min(6, time_steps)
    times_idx = np.linspace(0, time_steps-1, n_times, dtype=int)
    
    # Используем цветовую схему для разных времен
    colors_list = plt.cm.viridis(np.linspace(0, 1, n_times))
    
    for i, t_idx in enumerate(times_idx):
        ax3.plot(temperature[t_idx, :], color=colors_list[i], 
                linewidth=1.5, label=f't = {t_idx}')
    
    ax3.set_xlabel('Пространственная координата x', fontsize=11)
    ax3.set_ylabel('Температура', fontsize=11)
    ax3.set_title('Пространственное распределение\nв разные моменты времени', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Гистограмма распределения температур
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Создаем гистограмму
    counts, bins, patches = ax4.hist(temperature.flatten(), bins=50, 
                                      edgecolor='black', alpha=0.7, 
                                      color='steelblue')
    
    # Раскрашиваем гистограмму в соответствии с температурой
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    norm = plt.Normalize(vmin, vmax)
    for patch, bin_center in zip(patches, bin_centers):
        patch.set_facecolor(cmap(norm(bin_center)))
    
    ax4.set_xlabel('Температура', fontsize=11)
    ax4.set_ylabel('Частота', fontsize=11)
    ax4.set_title('Распределение температур', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Добавляем статистику
    stats_text = f'Min: {vmin:.4f}\nMax: {vmax:.4f}\nMean: {np.mean(temperature):.4f}\nStd: {np.std(temperature):.4f}'
    ax4.text(0.95, 0.95, stats_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.suptitle('Анализ решения уравнения теплопроводности', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Расширенная визуализация сохранена: {save_path}")
    
    plt.show()

def create_interactive_viewer(temperature, cmap_name='thermal'):
    """
    Создает интерактивное окно с ползунком времени
    """
    vmin, vmax = np.min(temperature), np.max(temperature)
    time_steps, space_points = temperature.shape
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(bottom=0.2)
    
    # Левая панель - профиль температуры в текущий момент времени
    current_time = 0
    line, = ax1.plot(temperature[current_time], 'r-', linewidth=2)
    ax1.set_xlim(0, space_points - 1)
    ax1.set_ylim(vmin - 0.05*(vmax-vmin), vmax + 0.05*(vmax-vmin))
    ax1.set_xlabel('Пространство x', fontsize=11)
    ax1.set_ylabel('Температура', fontsize=11)
    ax1.set_title(f'Профиль температуры при t = {current_time}', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.fill_between(range(space_points), 0, temperature[current_time], alpha=0.3, color='red')
    
    # Правая панель - тепловая карта с указанием текущего времени
    cmap = get_custom_colormap(cmap_name)
    im = ax2.imshow(temperature, aspect='auto', origin='lower', 
                    cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Добавляем горизонтальную линию для текущего времени
    time_line = ax2.axhline(y=current_time, color='white', linewidth=2, alpha=0.8, linestyle='--')
    ax2.set_xlabel('Пространство x', fontsize=11)
    ax2.set_ylabel('Время t', fontsize=11)
    ax2.set_title('Тепловая карта с индикатором времени', fontsize=12)
    
    # Цветовая шкала
    cbar = plt.colorbar(im, ax=ax2, label='Температура', fraction=0.046, pad=0.04)
    
    # Создаем слайдер
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Временной шаг', 0, time_steps - 1, 
                    valinit=0, valfmt='%d', valstep=1)
    
    def update(val):
        t = int(slider.val)
        line.set_ydata(temperature[t])
        ax1.set_title(f'Профиль температуры при t = {t}')
        
        # Обновляем заливку
        ax1.clear()
        ax1.plot(temperature[t], 'r-', linewidth=2)
        ax1.fill_between(range(space_points), 0, temperature[t], alpha=0.3, color='red')
        ax1.set_xlim(0, space_points - 1)
        ax1.set_ylim(vmin - 0.05*(vmax-vmin), vmax + 0.05*(vmax-vmin))
        ax1.set_xlabel('Пространство x', fontsize=11)
        ax1.set_ylabel('Температура', fontsize=11)
        ax1.set_title(f'Профиль температуры при t = {t}')
        ax1.grid(True, alpha=0.3)
        
        # Обновляем линию на тепловой карте
        time_line.set_ydata([t, t])
        
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    plt.suptitle('Интерактивный просмотр решения', fontsize=14, fontweight='bold')
    plt.show()

def create_animation(temperature, cmap_name='thermal', interval=50, save_path=None):
    """
    Создает анимацию распространения тепла во времени
    """
    vmin, vmax = np.min(temperature), np.max(temperature)
    time_steps, space_points = temperature.shape
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Левая панель - профиль температуры
    ax1.set_xlim(0, space_points - 1)
    ax1.set_ylim(vmin - 0.05*(vmax-vmin), vmax + 0.05*(vmax-vmin))
    ax1.set_xlabel('Пространство x', fontsize=11)
    ax1.set_ylabel('Температура', fontsize=11)
    ax1.set_title('Профиль температуры', fontsize=12)
    ax1.grid(True, alpha=0.3)
    line, = ax1.plot([], [], 'r-', linewidth=2)
    fill = ax1.fill_between([], [], alpha=0.3, color='red')
    
    # Правая панель - тепловая карта
    cmap = get_custom_colormap(cmap_name)
    im = ax2.imshow(temperature[0:1], aspect='auto', origin='lower', 
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    extent=[0, space_points-1, 0, time_steps-1])
    ax2.set_xlabel('Пространство x', fontsize=11)
    ax2.set_ylabel('Время t', fontsize=11)
    ax2.set_title('Тепловая карта', fontsize=12)
    cbar = plt.colorbar(im, ax=ax2, label='Температура', fraction=0.046, pad=0.04)
    
    time_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, 
                         fontsize=12, color='white', fontweight='bold',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    def animate(frame):
        # Обновляем профиль
        line.set_data(range(space_points), temperature[frame])
        
        # Обновляем заливку
        ax1.collections.clear()
        ax1.fill_between(range(space_points), 0, temperature[frame], 
                         alpha=0.3, color='red')
        
        # Обновляем тепловую карту (показываем до текущего времени)
        im.set_data(temperature[:frame+1])
        
        # Обновляем текст
        time_text.set_text(f't = {frame}')
        
        return [line, im, time_text]
    
    anim = animation.FuncAnimation(fig, animate, frames=time_steps, 
                                   interval=interval, blit=False, repeat=True)
    
    if save_path:
        try:
            anim.save(save_path, writer='ffmpeg', fps=20, dpi=100)
            print(f"✓ Анимация сохранена: {save_path}")
        except Exception as e:
            print(f"✗ Не удалось сохранить анимацию: {e}")
            print("  Установите ffmpeg: conda install ffmpeg или pip install ffmpeg-python")
    
    plt.show()
    return anim

# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================

def main():
    """
    Основная функция для запуска визуализации
    """
    print("\n" + "="*60)
    print("ТЕПЛОВАЯ ВИЗУАЛИЗАЦИЯ РЕШЕНИЯ УРАВНЕНИЯ ТЕПЛОПРОВОДНОСТИ")
    print("="*60 + "\n")
    
    # Запрос имени файла
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = input("Введите имя CSV файла с данными: ").strip()
    
    # Проверка существования файла
    if not os.path.exists(filename):
        print(f"\n✗ Файл '{filename}' не найден!")
        print("  Создайте файл с данными или укажите правильный путь.")
        sys.exit(1)
    
    # Загрузка данных
    temperature = load_heat_data(filename)
    
    # Меню выбора визуализации
    print("\nВыберите тип визуализации:")
    print("  1. Простая тепловая карта")
    print("  2. Расширенная визуализация (рекомендуется)")
    print("  3. Интерактивный просмотр")
    print("  4. Анимация")
    print("  5. Всё вместе")
    
    choice = input("\nВаш выбор (1-5): ").strip()
    
    # Цветовая схема
    print("\nВыберите цветовую схему:")
    print("  1. Thermal (синий->красный) - для тепловизора")
    print("  2. Jet (MATLAB style)")
    print("  3. Coolwarm (синий-белый-красный)")
    print("  4. Plasma (современная)")
    
    cmap_choice = input("\nВаш выбор (1-4): ").strip()
    cmap_names = {'1': 'thermal', '2': 'jet', '3': 'coolwarm', '4': 'plasma'}
    cmap_name = cmap_names.get(cmap_choice, 'thermal')
    
    # Сохранение результатов
    save = input("\nСохранить результаты? (y/n): ").strip().lower() == 'y'
    
    # Выполняем выбранную визуализацию
    if choice == '1':
        create_heatmap(temperature, cmap_name=cmap_name, 
                      save_path='heatmap.png' if save else None)
    
    elif choice == '2':
        create_enhanced_visualization(temperature, cmap_name=cmap_name,
                                     save_path='enhanced_viz.png' if save else None)
    
    elif choice == '3':
        create_interactive_viewer(temperature, cmap_name=cmap_name)
    
    elif choice == '4':
        create_animation(temperature, cmap_name=cmap_name,
                        save_path='animation.mp4' if save else None)
    
    elif choice == '5':
        create_heatmap(temperature, cmap_name=cmap_name, 
                      save_path='heatmap.png' if save else None)
        create_enhanced_visualization(temperature, cmap_name=cmap_name,
                                     save_path='enhanced_viz.png' if save else None)
        create_animation(temperature, cmap_name=cmap_name,
                        save_path='animation.mp4' if save else None)
    
    else:
        print("Неверный выбор. Используем расширенную визуализацию.")
        create_enhanced_visualization(temperature, cmap_name=cmap_name,
                                     save_path='enhanced_viz.png' if save else None)
    
    print("\n✓ Визуализация завершена!")

if __name__ == "__main__":
    main()