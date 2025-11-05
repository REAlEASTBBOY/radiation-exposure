"""
Модуль для расчета и визуализации поля облученности от прямоугольного источника света
на прямоугольном приемнике с использованием оптимизированных вычислений.

Использует:
- Numba для ускорения расчетов
- Matplotlib для интерактивной визуализации
- Физическую модель на основе закона обратных квадратов и косинусного распределения
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.colors import LogNorm, PowerNorm, Normalize
from matplotlib.patches import Rectangle
import time
from numba import jit, prange

# ============================================================================
# ОПТИМИЗИРОВАННАЯ ФУНКЦИЯ РАСЧЕТА ОБЛУЧЕННОСТИ
# ============================================================================

# Выносим расчетную функцию вне класса для работы с Numba
# Numba JIT компилятор ускоряет вычисления в 10-100 раз
@jit(nopython=True, parallel=True, fastmath=True)
def calculate_irradiance_optimized(X_r, Y_r, x_s_points, y_s_points, L_sr, I_i):
    """
    Оптимизированная версия расчета облученности с использованием Numba JIT.
    
    Физическая модель:
    - Использует закон обратных квадратов: E ∝ 1/r²
    - Учитывает косинус угла падения: E ∝ cos²(α)
    - Где α - угол между нормалью к поверхности и направлением на источник
    
    Параметры:
    ----------
    X_r, Y_r : numpy.ndarray
        Сетка координат точек приемника (meshgrid)
    x_s_points, y_s_points : numpy.ndarray
        Массивы координат точек источника
    L_sr : float
        Расстояние по оси Z между источником и приемником (м)
    I_i : float
        Интенсивность излучения на точку источника (Вт/ср)
    
    Возвращает:
    ----------
    result : numpy.ndarray
        Массив облученности для каждой точки приемника (Вт/м²)
    """
    # Определяем размер сетки приемника (квадратная сетка)
    match = X_r.shape[0]
    result = np.zeros((match, match))
    
    # Количество точек источника по осям X и Y
    num_x_s = len(x_s_points)
    num_y_s = len(y_s_points)
    
    # Параллельный цикл по точкам источника (ускорение через prange)
    # Для каждой точки источника вычисляем вклад во все точки приемника
    for i in prange(num_x_s):  # prange позволяет параллельное выполнение
        for j in range(num_y_s):
            # Координаты текущей точки источника
            x_s = x_s_points[i]
            y_s = y_s_points[j]
            
            # Вектор от точки источника к каждой точке приемника
            # dx и dy - это массивы разностей координат для всех точек приемника
            dx = X_r - x_s
            dy = Y_r - y_s
            
            # Квадрат расстояния в 3D пространстве
            # L_sr² - это расстояние по оси Z (вертикальное расстояние)
            distance_sq = dx**2 + dy**2 + L_sr**2
            
            # Косинус угла между нормалью к приемнику и направлением на источник
            # cos(α) = L_sr / sqrt(dx² + dy² + L_sr²)
            # Это соответствует косинусному распределению излучения
            cos_alpha = L_sr / np.sqrt(distance_sq)
            
            # Расчет облученности по физической модели:
            # E = I_i * cos²(α) / r²
            # где cos²(α) учитывает закон Ламбера для косинусного источника
            irradiance = I_i * (cos_alpha**2) / distance_sq
            
            # Суммируем вклад от всех точек источника
            result += irradiance
    
    return result

# ============================================================================
# КЛАСС ИНТЕРАКТИВНОГО ПРИЛОЖЕНИЯ
# ============================================================================

class InteractiveIrradianceApp:
    """
    Класс для создания интерактивного приложения визуализации поля облученности.
    
    Предоставляет графический интерфейс с:
    - Интерактивными слайдерами для изменения параметров
    - Визуализацией схемы расположения источника и приемника
    - Отображением поля облученности в реальном времени
    - Различными режимами нормализации данных
    """
    
    def __init__(self):
        """
        Инициализация приложения с начальными параметрами системы.
        
        Параметры системы:
        ----------
        p : float
            Мощность источника излучения (Вт)
        l_r, h_r : float
            Размеры приемника: длина и высота (м)
        x, y : float
            Координаты центра источника в плоскости XY (м)
        R : float
            Расстояние по оси Z между источником и приемником (м)
        l_s, h_s : float
            Размеры источника: длина и высота (м)
        accuracy : int
            Точность расчета (количество точек разбиения сетки)
        """
        # ============ ПАРАМЕТРЫ СИСТЕМЫ ============
        self.p = 500  # Мощность источника излучения (Вт)
        self.l_r = 100  # Длина приемника по оси X (м)
        self.h_r = 100  # Высота приемника по оси Y (м)
        self.x = 50  # Позиция центра источника по X (м)
        self.y = 50  # Позиция центра источника по Y (м)
        self.R = 500  # Расстояние Z (L_sr) между плоскостями (м)
        self.l_s = 1  # Длина источника по оси X (м)
        self.h_s = 1  # Высота источника по оси Y (м)
        self.accuracy = 30  # Точность расчета (размер сетки: accuracy × accuracy точек)
        
        # ============ ПАРАМЕТРЫ ВИЗУАЛИЗАЦИИ ============
        self.normalization_type = 'linear'  # Тип нормализации: 'linear', 'log', 'power'
        self.power_gamma = 0.5  # Параметр gamma для power normalization
        
        # ============ ПЕРЕМЕННЫЕ ДЛЯ ХРАНЕНИЯ РЕЗУЛЬТАТОВ ============
        self.min_irradiance = 0  # Минимальная облученность (Вт/м²)
        self.max_irradiance = 0  # Максимальная облученность (Вт/м²)
        self.result_array = None  # Массив результатов расчета
        self.calc_time = 0  # Время последнего расчета (сек)
        
        # Инициализация интерфейса и первичный расчет
        self.setup_ui()
        self.update()
        
    def setup_ui(self):
        """Настройка интерфейса"""
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.suptitle('Интерактивная визуализация поля облученности (оптимизированная)', fontsize=14, fontweight='bold')
        
        # Левая панель - схема расположения
        self.ax_layout = plt.subplot(1, 2, 1)
        
        # Правая панель - поле облученности
        self.ax_field = plt.subplot(1, 2, 2)
        
        self.create_sliders()
        self.create_control_buttons()
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.4, right=0.98)
    
    def calculate_irradiance_extended_model(self):
        """
        Улучшенная модель расчета облученности с учетом:
        - Закона обратных квадратов для больших расстояний
        - Поправок для малых расстояний
        - Оптимизации вычислений
        
        Алгоритм:
        1. Создает сетку точек приемника (равномерное разбиение)
        2. Определяет точки источника (адаптивное количество)
        3. Распределяет мощность равномерно по точкам источника
        4. Вычисляет вклад каждой точки источника во все точки приемника
        5. Применяет поправку для малых расстояний
        
        Возвращает:
        ----------
        numpy.ndarray
            Массив облученности размером (accuracy × accuracy)
        """
        L_sr = self.R
        match = self.accuracy
        
        # Создаем сетку точек приемника
        x_r = np.linspace(0, self.l_r, match)
        y_r = np.linspace(0, self.h_r, match)
        X_r, Y_r = np.meshgrid(x_r, y_r, indexing='xy')
        
        # Адаптивное количество точек источника в зависимости от размера
        source_points_factor = max(2, min(10, match // 10))
        x_s_points = np.linspace(-self.l_s/2, self.l_s/2, source_points_factor) + self.x
        y_s_points = np.linspace(-self.h_s/2, self.h_s/2, source_points_factor) + self.y
        
        # Общая мощность распределяется равномерно по всем точкам источника
        num_source_points = len(x_s_points) * len(y_s_points)
        power_per_point = self.p / num_source_points
        
        # Интенсивность на точку источника
        I_i = power_per_point / np.pi
        
        # Расчет облученности с использованием оптимизированной функции
        result = calculate_irradiance_optimized(X_r, Y_r, x_s_points, y_s_points, L_sr, I_i)
        
        # Поправка для малых расстояний (когда расстояние сравнимо с размерами)
        if L_sr < max(self.l_s, self.h_s) * 10:
            # Эмпирическая поправка для учета конечных размеров источника
            size_correction = 1.0 + 0.1 * (max(self.l_s, self.h_s) / L_sr)
            result *= size_correction
        
        return result
    
    def create_control_buttons(self):
        """Создание кнопок управления визуализацией"""
        # Кнопки выбора типа нормализации
        ax_radio = plt.axes([0.65, 0.25, 0.1, 0.1])
        self.radio_norm = RadioButtons(ax_radio, ['linear', 'log', 'power'])
        self.radio_norm.on_clicked(self.on_norm_change)
        
        # Слайдер для gamma correction
        ax_gamma = plt.axes([0.65, 0.15, 0.2, 0.02])
        self.slider_gamma = Slider(ax_gamma, 'Gamma', 0.1, 2.0, valinit=0.5)
        self.slider_gamma.on_changed(self.on_gamma_change)
        
        # Кнопка сброса
        ax_reset = plt.axes([0.65, 0.05, 0.1, 0.04])
        self.button_reset = Button(ax_reset, 'Сброс')
        self.button_reset.on_clicked(self.on_reset)
        
        # Кнопка быстрого обновления
        ax_fast = plt.axes([0.77, 0.05, 0.1, 0.04])
        self.button_fast = Button(ax_fast, 'Быстрый режим')
        self.button_fast.on_clicked(self.on_fast_mode)
    
    def on_norm_change(self, label):
        self.normalization_type = label
        self.update_display()
    
    def on_gamma_change(self, val):
        self.power_gamma = val
        if self.normalization_type == 'power':
            self.update_display()
    
    def on_reset(self, event):
        self.slider_power.reset()
        self.slider_x.reset()
        self.slider_y.reset()
        self.slider_z.reset()
        self.slider_ls.reset()
        self.slider_hs.reset()
        self.slider_lr.reset()
        self.slider_hr.reset()
        self.slider_acc.reset()
        self.slider_gamma.reset()
        self.radio_norm.set_active(0)
    
    def on_fast_mode(self, event):
        """Включение быстрого режима с меньшей точностью"""
        self.sliders['slider_acc'].set_val(30)
        print("⚡ Включен быстрый режим (точность: 30)")
    
    def create_sliders(self):
        """Создание слайдеров с увеличенным диапазоном расстояний"""
        slider_y_start = 0.02
        slider_height = 0.02
        slider_spacing = 0.03
        
        sliders_config = [
            ('Мощность (Вт)', 'slider_power', 1, 5000, self.p, 10),
            ('Позиция X (м)', 'slider_x', 0, 200, self.x, 1),
            ('Позиция Y (м)', 'slider_y', 0, 200, self.y, 1),
            ('Расстояние Z (м)', 'slider_z', 200, 1000, self.R, 10),  # Увеличенный диапазон
            ('Длина источника (м)', 'slider_ls', 0.1, 20, self.l_s, 0.1),
            ('Высота источника (м)', 'slider_hs', 0.1, 20, self.h_s, 0.1),
            ('Длина приемника (м)', 'slider_lr', 10, 500, self.l_r, 10),
            ('Высота приемника (м)', 'slider_hr', 10, 500, self.h_r, 10),
            ('Точность', 'slider_acc', 20, 80, self.accuracy, 5)  # Оптимальный диапазон
        ]
        
        self.sliders = {}
        for i, (label, name, vmin, vmax, valinit, valstep) in enumerate(sliders_config):
            ax = plt.axes([0.15, slider_y_start + (8-i)*slider_spacing, 0.3, slider_height])
            slider = Slider(ax, label, vmin, vmax, valinit=valinit, valstep=valstep)
            slider.on_changed(self.on_slider_change)
            self.sliders[name] = slider
    
    def on_slider_change(self, val):
        """Обновление параметров с проверкой производительности"""
        self.p = self.sliders['slider_power'].val
        self.x = self.sliders['slider_x'].val
        self.y = self.sliders['slider_y'].val
        self.R = self.sliders['slider_z'].val
        self.l_s = self.sliders['slider_ls'].val
        self.h_s = self.sliders['slider_hs'].val
        self.l_r = self.sliders['slider_lr'].val
        self.h_r = self.sliders['slider_hr'].val
        new_accuracy = int(self.sliders['slider_acc'].val)
        
        # Автоматическое обновление только если точность не слишком высокая
        if new_accuracy <= 50:  # Ограничение для быстрого отклика
            self.accuracy = new_accuracy
            self.update()
        else:
            # Для высокой точности показываем предупреждение
            self.accuracy = new_accuracy
            print("⚠️  Высокая точность - расчет может занять время...")
            self.update()
    
    def update(self):
        """
        Полное обновление всех визуализаций с измерением времени расчета.
        
        Выполняет:
        1. Расчет поля облученности
        2. Определение минимальных и максимальных значений
        3. Обновление схемы расположения
        4. Обновление поля облученности
        5. Вывод информации о расчете в консоль
        """
        start_time = time.time()
        
        # Расчет поля облученности
        result_array = self.calculate_irradiance_extended_model()
        self.result_array = result_array
        
        # Сохранение статистики облученности
        self.min_irradiance = np.min(result_array)
        self.max_irradiance = np.max(result_array)
        self.calc_time = time.time() - start_time
        
        # Обновление визуализаций
        self.update_layout()      # Левая панель: схема расположения
        self.update_display()     # Правая панель: поле облученности
        
        # Вывод информации о производительности
        self.print_calculation_info()
    
    def update_layout(self):
        """Обновление схемы расположения"""
        self.ax_layout.clear()
        self.ax_layout.set_title('Расположение источника и приемника (вид сверху)', fontsize=12)
        self.ax_layout.set_xlabel('X координата (м)')
        self.ax_layout.set_ylabel('Y координата (м)')
        self.ax_layout.grid(True, alpha=0.3)
        self.ax_layout.set_aspect('equal')
        
        # Приемник
        receiver_rect = Rectangle((0, 0), self.l_r, self.h_r, 
                                 linewidth=2, edgecolor='blue', 
                                 facecolor='lightblue', alpha=0.3, label='Приемник')
        self.ax_layout.add_patch(receiver_rect)
        
        # Источник (центрированный)
        source_x = self.x - self.l_s/2
        source_y = self.y - self.h_s/2
        source_rect = Rectangle((source_x, source_y), self.l_s, self.h_s, 
                               linewidth=2, edgecolor='red', 
                               facecolor='lightcoral', alpha=0.7, label='Источник')
        self.ax_layout.add_patch(source_rect)
        
        # Центры
        self.ax_layout.plot(self.l_r/2, self.h_r/2, 'bo', markersize=8, label='Центр приемника')
        self.ax_layout.plot(self.x, self.y, 'ro', markersize=8, label='Центр источника')
        
        # Линия соединения
        self.ax_layout.plot([self.x, self.l_r/2], [self.y, self.h_r/2], 'k--', alpha=0.5, linewidth=1)
        
        # Границы
        margin = 20
        self.ax_layout.set_xlim(-margin, max(self.l_r, self.x + self.l_s/2) + margin)
        self.ax_layout.set_ylim(-margin, max(self.h_r, self.y + self.h_s/2) + margin)
        
        self.ax_layout.legend(loc='upper right')
        
        # Детальная информация с временем расчета
        info_text = (f'Параметры системы:\n'
                    f'• Мощность: {self.p} Вт\n'
                    f'• Расстояние Z: {self.R} м\n'
                    f'• Размер источника: {self.l_s:.1f}×{self.h_s:.1f} м\n'
                    f'• Размер приемника: {self.l_r}×{self.h_r} м\n'
                    f'• Точек расчета: {self.accuracy}×{self.accuracy}\n'
                    f'• Время расчета: {self.calc_time:.3f} сек\n\n'
                    f'Облученность:\n'
                    f'• МАКСИМАЛЬНАЯ: {self.max_irradiance:.2e} Вт/м²\n'
                    f'• минимальная: {self.min_irradiance:.2e} Вт/м²')
        
        if self.min_irradiance > 0:
            ratio = self.max_irradiance / self.min_irradiance
            info_text += f'\n• Отношение: {ratio:.1f}'
        
        self.ax_layout.text(0.02, 0.98, info_text, transform=self.ax_layout.transAxes,
                           verticalalignment='top', 
                           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
                           fontsize=9, fontfamily='monospace')
    
    def update_display(self):
        """Обновление отображения поля облученности"""
        self.ax_field.clear()
        self.ax_field.set_title(f'Поле облученности приемника (расчет: {self.calc_time:.3f} сек)', fontsize=12)
        self.ax_field.set_xlabel('X координата приемника (м)')
        self.ax_field.set_ylabel('Y координата приемника (м)')
        
        result_array = self.result_array
        
        # Выбор нормализации
        if self.normalization_type == 'log':
            norm = LogNorm(vmin=max(1e-10, self.min_irradiance), vmax=self.max_irradiance)
            cmap = 'viridis'
            cbar_label = 'Облученность (Вт/м²) - лог. шкала'
        elif self.normalization_type == 'power':
            norm = PowerNorm(gamma=self.power_gamma, vmin=self.min_irradiance, vmax=self.max_irradiance)
            cmap = 'plasma'
            cbar_label = 'Облученность (Вт/м²)'
        else:
            norm = Normalize(vmin=self.min_irradiance, vmax=self.max_irradiance)
            cmap = 'hot'
            cbar_label = 'Облученность (Вт/м²)'
        
        # Отображение
        im = self.ax_field.imshow(result_array, cmap=cmap, aspect='auto', 
                                 origin='lower', interpolation='bilinear',
                                 extent=[0, self.l_r, 0, self.h_r], norm=norm)
        
        # Colorbar
        if hasattr(self, 'cbar'):
            self.cbar.remove()
        self.cbar = plt.colorbar(im, ax=self.ax_field, label=cbar_label)
        
        # Контуры только для умеренной точности (для скорости)
        if self.accuracy <= 50 and self.max_irradiance > self.min_irradiance and self.min_irradiance > 0:
            if self.normalization_type == 'log':
                levels = np.logspace(np.log10(self.min_irradiance), np.log10(self.max_irradiance), 6)
            else:
                levels = np.linspace(self.min_irradiance, self.max_irradiance, 6)
            
            contour = self.ax_field.contour(result_array, levels=levels[1:-1], 
                                          colors='white', alpha=0.5, linewidths=0.8,
                                          extent=[0, self.l_r, 0, self.h_r])
        
        # Информация об облученности
        info_text = (f'ОБЛУЧЕННОСТЬ:\n'
                    f'МАКС: {self.max_irradiance:.2e} Вт/м²\n'
                    f'мин: {self.min_irradiance:.2e} Вт/м²')
        
        if self.min_irradiance > 0:
            ratio = self.max_irradiance / self.min_irradiance
            info_text += f'\nотношение: {ratio:.0f}'
            
            max_pos = np.unravel_index(np.argmax(result_array), result_array.shape)
            max_x = max_pos[1] * (self.l_r / (self.accuracy - 1)) if self.accuracy > 1 else 0
            max_y = max_pos[0] * (self.h_r / (self.accuracy - 1)) if self.accuracy > 1 else 0
            info_text += f'\nмакс в: ({max_x:.1f}, {max_y:.1f}) м'
        
        self.ax_field.text(0.02, 0.98, info_text, transform=self.ax_field.transAxes,
                          verticalalignment='top', fontsize=10, fontweight='bold',
                          bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
                          color='white', fontfamily='monospace')
        
        # Информация о настройках
        norm_info = f'Нормализация: {self.normalization_type}'
        if self.normalization_type == 'power':
            norm_info += f' (γ={self.power_gamma:.1f})'
        
        self.ax_field.text(0.02, 0.02, norm_info, transform=self.ax_field.transAxes,
                          verticalalignment='bottom', fontsize=9,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        self.fig.canvas.draw_idle()

    def print_calculation_info(self):
        """Вывод информации о расчете"""
        performance = "⚡ Быстро" if self.calc_time < 0.1 else "⏱️  Нормально" if self.calc_time < 0.5 else "🐢 Медленно"
        print(f"{performance} | Время: {self.calc_time:.3f} сек | "
              f"Точки: {self.accuracy}×{self.accuracy} | "
              f"Облученность: {self.min_irradiance:.2e} - {self.max_irradiance:.2e} Вт/м²")

# Запуск
if __name__ == '__main__':
    print("🚀 Запуск оптимизированной версии с улучшенной моделью расчета...")
    print("📏 Диапазон расстояний: 200-1000 м")
    print("⚡ Используется Numba для ускорения расчетов")
    print("💡 Советы: Используйте точность 20-40 для быстрого отклика")
    
    app = InteractiveIrradianceApp()
    plt.show()