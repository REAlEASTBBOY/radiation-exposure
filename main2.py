import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

def calculate_irradiance_vectorized(p, l_r=100, h_r=100, x=0, y=0, R=30, 
                                   l_s=1, h_s=10, accuracy=20):
    """
    Векторизованная версия расчета облученности
    """
    L_sr = R
    match = accuracy
    
    # Создаем сетку точек приемника
    x_r = np.linspace(0, float(l_r), match)
    y_r = np.linspace(0, float(h_r), match)
    X_r, Y_r = np.meshgrid(x_r, y_r)
    
    # Создаем сетку точек источника
    x_s = np.linspace(0, float(l_s), match) + float(x)
    y_s = np.linspace(0, float(h_s), match) + float(y)
    X_s, Y_s = np.meshgrid(x_s, y_s)
    
    P_source = p
    I_i = P_source / (np.pi * match**2)
    
    # Векторизованный расчет
    result = np.zeros((match, match))
    
    # Для каждой точки источника суммируем вклад во все точки приемника
    for i in range(match):
        for j in range(match):
            dx = X_r - x_s[i]
            dy = Y_r - y_s[j]
            L_vect = np.sqrt(dx**2 + dy**2 + L_sr**2)
            cos_a = L_sr / L_vect
            result += I_i * cos_a**2 / L_sr**2
    
    return result / 0.005

def main_vectorized(p: int, l_r: str = "20", h_r: str = "20", x="20", y="20", 
                   R: int = 30, l_s: str = "1", h_s: str = "1", accuracy: int = 10):
    """
    Основная функция с векторизованными вычислениями
    """
    result_array = calculate_irradiance_vectorized(
        p, float(l_r), float(h_r), float(x), float(y), R, 
        float(l_s), float(h_s), accuracy
    )
    
    # Визуализация (остается без изменений)
    min_irradiance = np.min(result_array)
    max_irradiance = np.max(result_array)
    print(f"Max result: {max_irradiance}")
    
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
    
    plt.figure(figsize=(float(l_r), float(h_r)))
    im = plt.imshow(result_array, cmap='coolwarm', aspect='auto', origin='lower', 
                    interpolation='bilinear', extent=[0, float(h_r), 0, float(l_r)],
                    norm=norm)
    plt.colorbar(im, label='Облученность нормализированная')
    plt.xlabel('Y координата приемника (м)')
    plt.ylabel('X координата приемника (м)')
    plt.title('Поле облученности приемника')
    plt.grid(True, alpha=0.3)
    
    plt.text(0.02, 0.98, f'Min: {min_irradiance:.3e} Вт/м²\nMax: {max_irradiance:.3e} Вт/м²',
            transform=plt.gca().transAxes, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10)
    
    plt.show()

# Запуск
main_vectorized(50)