import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import time

class InteractiveIrradianceApp:
    def __init__(self):
        # Начальные параметры
        self.p = 50  # Мощность (Вт)
        self.l_r = 100  # Длина приемника (м)
        self.h_r = 100  # Высота приемника (м)
        self.x = 50  # Позиция источника X (м)
        self.y = 50  # Позиция источника Y (м)
        self.R = 30  # Расстояние Z (L_sr) (м)
        self.l_s = 1  # Длина источника (м)
        self.h_s = 10  # Высота источника (м)
        self.accuracy = 20  # Точность расчета
        
        # Создаем фигуру с двумя подграфиками
        self.fig = plt.figure(figsize=(16, 8))
        self.fig.suptitle('Интерактивная визуализация поля облученности', fontsize=14, fontweight='bold')
        
        # Левая панель - схема расположения
        self.ax_layout = plt.subplot(1, 2, 1)
        self.ax_layout.set_title('Расположение источника и приемника (вид сверху)')
        self.ax_layout.set_xlabel('X (м)')
        self.ax_layout.set_ylabel('Y (м)')
        self.ax_layout.grid(True, alpha=0.3)
        self.ax_layout.set_aspect('equal')
        
        # Правая панель - поле облученности
        self.ax_field = plt.subplot(1, 2, 2)
        self.ax_field.set_title('Поле облученности приемника')
        self.ax_field.set_xlabel('Y координата приемника (м)')
        self.ax_field.set_ylabel('X координата приемника (м)')
        
        # Создаем слайдеры
        self.create_sliders()
        
        # Первоначальный расчет и отображение
        self.update()
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.35, right=0.98)
        
    def calculate_irradiance_vectorized(self):
        """Векторизованная версия расчета облученности"""
        L_sr = self.R
        match = self.accuracy
        
        # Создаем сетку точек приемника
        x_r = np.linspace(0, self.l_r, match)
        y_r = np.linspace(0, self.h_r, match)
        X_r, Y_r = np.meshgrid(x_r, y_r)
        
        # Создаем сетку точек источника
        x_s = np.linspace(0, self.l_s, match) + self.x
        y_s = np.linspace(0, self.h_s, match) + self.y
        
        P_source = self.p
        I_i = P_source / (np.pi * match**2)
        
        # Векторизованный расчет
        result = np.zeros((match, match))
        
        # Для каждой точки источника суммируем вклад во все точки приемника
        # i - индекс по Y (строки), j - индекс по X (столбцы)
        for i in range(match):
            for j in range(match):
                dx = X_r - x_s[j]  # X координата источника
                dy = Y_r - y_s[i]  # Y координата источника
                L_vect = np.sqrt(dx**2 + dy**2 + L_sr**2)
                cos_a = L_sr / L_vect
                result += I_i * cos_a**2 / L_sr**2
        
        return result / 0.005
    
    def create_sliders(self):
        """Создание слайдеров для управления параметрами"""
        # Позиции слайдеров
        slider_y_start = 0.02
        slider_height = 0.02
        slider_spacing = 0.03
        
        # Мощность источника
        ax_power = plt.axes([0.15, slider_y_start + 8*slider_spacing, 0.3, slider_height])
        self.slider_power = Slider(ax_power, 'Мощность (Вт)', 1, 500, valinit=self.p, valstep=1)
        self.slider_power.on_changed(self.on_slider_change)
        
        # Позиция источника X
        ax_x = plt.axes([0.15, slider_y_start + 7*slider_spacing, 0.3, slider_height])
        self.slider_x = Slider(ax_x, 'Позиция X (м)', 0, 200, valinit=self.x, valstep=1)
        self.slider_x.on_changed(self.on_slider_change)
        
        # Позиция источника Y
        ax_y = plt.axes([0.15, slider_y_start + 6*slider_spacing, 0.3, slider_height])
        self.slider_y = Slider(ax_y, 'Позиция Y (м)', 0, 200, valinit=self.y, valstep=1)
        self.slider_y.on_changed(self.on_slider_change)
        
        # Расстояние Z (L_sr)
        ax_z = plt.axes([0.15, slider_y_start + 5*slider_spacing, 0.3, slider_height])
        self.slider_z = Slider(ax_z, 'Расстояние Z (м)', 1, 200, valinit=self.R, valstep=1)
        self.slider_z.on_changed(self.on_slider_change)
        
        # Размер источника - длина
        ax_ls = plt.axes([0.15, slider_y_start + 4*slider_spacing, 0.3, slider_height])
        self.slider_ls = Slider(ax_ls, 'Длина источника (м)', 0.1, 50, valinit=self.l_s, valstep=0.1)
        self.slider_ls.on_changed(self.on_slider_change)
        
        # Размер источника - высота
        ax_hs = plt.axes([0.15, slider_y_start + 3*slider_spacing, 0.3, slider_height])
        self.slider_hs = Slider(ax_hs, 'Высота источника (м)', 0.1, 50, valinit=self.h_s, valstep=0.1)
        self.slider_hs.on_changed(self.on_slider_change)
        
        # Размер приемника - длина
        ax_lr = plt.axes([0.15, slider_y_start + 2*slider_spacing, 0.3, slider_height])
        self.slider_lr = Slider(ax_lr, 'Длина приемника (м)', 10, 200, valinit=self.l_r, valstep=1)
        self.slider_lr.on_changed(self.on_slider_change)
        
        # Размер приемника - высота
        ax_hr = plt.axes([0.15, slider_y_start + 1*slider_spacing, 0.3, slider_height])
        self.slider_hr = Slider(ax_hr, 'Высота приемника (м)', 10, 200, valinit=self.h_r, valstep=1)
        self.slider_hr.on_changed(self.on_slider_change)
        
        # Точность расчета
        ax_acc = plt.axes([0.15, slider_y_start + 0*slider_spacing, 0.3, slider_height])
        self.slider_acc = Slider(ax_acc, 'Точность', 5, 50, valinit=self.accuracy, valstep=1, valfmt='%d')
        self.slider_acc.on_changed(self.on_slider_change)
        
        # Кнопка обновления
        ax_button = plt.axes([0.55, slider_y_start + 1*slider_spacing, 0.1, 0.04])
        self.button_update = Button(ax_button, 'Обновить')
        self.button_update.on_clicked(self.on_button_update)
        
        # Переменные для хранения графических элементов
        self.receiver_rect = None
        self.source_rect = None
        self.im_field = None
        self.cbar = None
        self.info_text = None
    
    def on_slider_change(self, val):
        """Обновление параметров при изменении слайдеров"""
        self.p = int(self.slider_power.val)
        self.x = self.slider_x.val
        self.y = self.slider_y.val
        self.R = self.slider_z.val
        self.l_s = self.slider_ls.val
        self.h_s = self.slider_hs.val
        self.l_r = self.slider_lr.val
        self.h_r = self.slider_hr.val
        self.accuracy = int(self.slider_acc.val)
        
        # Обновляем отображение
        self.update()
    
    def on_button_update(self, event):
        """Принудительное обновление по кнопке"""
        self.update()
    
    def update(self):
        """Обновление всех графиков"""
        start_time = time.time()
        
        # Расчет поля облученности
        result_array = self.calculate_irradiance_vectorized()
        
        # Обновление схемы расположения
        self.update_layout()
        
        # Обновление поля облученности
        self.update_field(result_array)
        
        calc_time = time.time() - start_time
        print(f"Расчет выполнен за {calc_time:.2f} сек")
    
    def update_layout(self):
        """Обновление схемы расположения источника и приемника"""
        self.ax_layout.clear()
        self.ax_layout.set_title('Расположение источника и приемника (вид сверху)')
        self.ax_layout.set_xlabel('X (м)')
        self.ax_layout.set_ylabel('Y (м)')
        self.ax_layout.grid(True, alpha=0.3)
        self.ax_layout.set_aspect('equal')
        
        # Приемник (синий прямоугольник)
        receiver_rect = Rectangle((0, 0), self.l_r, self.h_r, 
                                 linewidth=2, edgecolor='blue', 
                                 facecolor='lightblue', alpha=0.3, label='Приемник')
        self.ax_layout.add_patch(receiver_rect)
        
        # Источник (красный прямоугольник)
        source_rect = Rectangle((self.x, self.y), self.l_s, self.h_s, 
                               linewidth=2, edgecolor='red', 
                               facecolor='lightcoral', alpha=0.5, label='Источник')
        self.ax_layout.add_patch(source_rect)
        
        # Линия, показывающая расстояние Z
        self.ax_layout.plot([self.x + self.l_s/2, self.l_r/2], 
                           [self.y + self.h_s/2, self.h_r/2], 
                           'k--', linewidth=1, alpha=0.5, label=f'Z={self.R}м')
        
        # Установка границ
        margin = 20
        self.ax_layout.set_xlim(-margin, max(self.l_r, self.x + self.l_s) + margin)
        self.ax_layout.set_ylim(-margin, max(self.h_r, self.y + self.h_s) + margin)
        
        # Легенда
        self.ax_layout.legend(loc='upper right')
        
        # Информация о параметрах
        info = f'Мощность: {self.p} Вт\nZ: {self.R} м\nТочность: {self.accuracy}'
        self.ax_layout.text(0.02, 0.98, info, transform=self.ax_layout.transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round', 
                           facecolor='wheat', alpha=0.8), fontsize=9)
    
    def update_field(self, result_array):
        """Обновление поля облученности"""
        self.ax_field.clear()
        self.ax_field.set_title('Поле облученности приемника')
        self.ax_field.set_xlabel('X координата приемника (м) - длина')
        self.ax_field.set_ylabel('Y координата приемника (м) - высота')
        
        # Нормализация
        min_irradiance = np.min(result_array)
        max_irradiance = np.max(result_array)
        
        vmin = min_irradiance
        vmax = max_irradiance
        
        if vmin <= 1 <= vmax:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
        elif vmax < 1:
            vmax_sym = 2 - vmin
            norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax_sym)
        elif vmin > 1:
            vmin_sym = 2 - vmax
            norm = TwoSlopeNorm(vmin=vmin_sym, vcenter=1.0, vmax=vmax)
        else:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
        
        # Отображение поля
        # extent = [x_min, x_max, y_min, y_max]
        # l_r - длина по X, h_r - высота по Y
        # Вычисляем aspect ratio для сохранения пропорций
        aspect_ratio = self.h_r / self.l_r if self.l_r > 0 else 1.0
        
        # Устанавливаем adjustable для сохранения пропорций
        self.ax_field.set_aspect(aspect_ratio, adjustable='box')
        
        im = self.ax_field.imshow(result_array, cmap='coolwarm', aspect=aspect_ratio, 
                                 origin='lower', interpolation='bilinear',
                                 extent=[0, self.l_r, 0, self.h_r], norm=norm)
        
        # Colorbar
        if self.cbar is not None:
            self.cbar.remove()
        self.cbar = plt.colorbar(im, ax=self.ax_field, label='Облученность нормализированная')
        
        # Сетка
        self.ax_field.grid(True, alpha=0.3)
        
        # Информация о min/max
        info_text = f'Min: {min_irradiance:.3e} Вт/м²\nMax: {max_irradiance:.3e} Вт/м²'
        self.ax_field.text(0.02, 0.98, info_text, transform=self.ax_field.transAxes,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                          fontsize=10)
        
        self.fig.canvas.draw_idle()

# Запуск приложения
if __name__ == '__main__':
    app = InteractiveIrradianceApp()
    plt.show()
