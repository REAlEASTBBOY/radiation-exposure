import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from decimal import Decimal, getcontext
import asyncio




async def ps_from_point(x, y, xs,ys, x_pos, y_pos,L_sr,I_i):
    xs_actual = xs + x_pos
    ys_actual = ys + y_pos
    dx = float(xs_actual - x)
    dy = float(ys_actual - y)
    L_vect = np.sqrt(dx**2 + dy**2 + (L_sr)**2)
    cos_a = L_sr/L_vect
    return  I_i*cos_a**2/L_sr**2



async def ps_for_target(x, y, l_s,h_s, xstep_s, ystep_s, x_pos,y_pos,L_sr,I_i):
    tasks_list = []
    xs = Decimal('0')
    while xs < l_s:
        ys = Decimal('0')
        while ys < h_s:
            task = ps_from_point(x, y, xs, ys, x_pos, y_pos, L_sr, I_i)
            tasks_list.append(task)
            ys += ystep_s
        xs += xstep_s
    
    # Выполняем все задачи и суммируем результаты
    results = await asyncio.gather(*tasks_list)
    return sum(results)

async def ps_for_all(match,l_r,h_r,l_s,h_s, x_pos,y_pos,L_sr,I_i):
    result = [[0]* match for _ in range(match)]
    x_step_r = l_r / Decimal(match)
    y_step_r = h_r / Decimal(match)
    xstep_s = l_s/ Decimal(match)
    ystep_s = h_s/Decimal(match)
    
    # Собираем все задачи
    tasks_list = []
    xi = 0
    x = Decimal('0')
    while x < l_r and xi < match:
        yi = 0
        y = Decimal('0')
        while y < h_r and yi < match:
            task = ps_for_target(x, y, l_s, h_s, xstep_s, ystep_s, x_pos, y_pos, L_sr, I_i)
            tasks_list.append((task, xi, yi))
            yi += 1
            y += y_step_r
        xi += 1
        x += x_step_r
    
    # Выполняем все задачи
    for task, xi, yi in tasks_list:
        result[xi][yi] = await task
    
    print(f"Результат: {len(result)}x{len(result[0])}")
    return result
    




async def main(p:int,l_r:str = "100", h_r:str = "100",x="0", y="100", R:int = 30,l_s:str = "1", h_s:str="10",accuracy: int = 10):
    getcontext().prec = 28
    L_sr = R #м
    match = accuracy
    #Размер косинусного источника света прямоугольник 
    a = Decimal(l_s) #м
    b = Decimal(h_s) #м
    x_pos = Decimal(x)
    y_pos = Decimal(y)
    #Размер приемника света прямоугольник
    c = Decimal(l_r) #м
    d = Decimal(h_r) #м
    P_source = p #Вт
    I_i = P_source/(np.pi * match**2)#Вт/ср
    result_array = await ps_for_all(match,c,d,a,b, x_pos, y_pos, L_sr, I_i)
    # Преобразуем результат в numpy массив для удобства работы
    result_array = np.array(result_array)/0.005 # максимальная плотность мощности при 10м и 10 вт 
    #нормализируем


    # Находим минимальную и максимальную облученность
    min_irradiance = np.min(result_array)
    max_irradiance = np.max(result_array)
    print(np.max(result_array))

    # Нормализуем значения для colormap так, чтобы значение 1 всегда было нейтральной точкой (белый)
    vmin = min_irradiance
    vmax = max_irradiance

    # Всегда используем vcenter=1.0 для TwoSlopeNorm, чтобы значение 1 было нейтральной точкой (белый)
    # Расширяем диапазон симметрично относительно 1, если нужно
    if vmin <= 1 <= vmax:
        # Значение 1 находится в диапазоне, используем как есть
        norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    elif vmax < 1:
        # Все значения < 1, расширяем диапазон симметрично относительно 1
        # Максимальное расстояние от 1 = 1 - vmin (самое дальнее значение)
        # Для симметрии расширяем вправо: vmax_sym = 1 + (1 - vmin) = 2 - vmin
        vmax_sym = 2 - vmin
        norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax_sym)
    elif vmin > 1:
        # Все значения > 1, расширяем диапазон симметрично относительно 1
        # Максимальное расстояние от 1 = vmax - 1 (самое дальнее значение)
        # Для симметрии расширяем влево: vmin_sym = 1 - (vmax - 1) = 2 - vmax
        vmin_sym = 2 - vmax
        norm = TwoSlopeNorm(vmin=vmin_sym, vcenter=1.0, vmax=vmax)
    else:
        # На всякий случай
        norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    
    # Используем стандартную diverging colormap
    colormap = 'coolwarm'

    # Создаем визуализацию
    plt.figure(figsize=(float(c), float(d)))

    # Создаем цветовую карту поля облученности
    # Используем стандартную diverging colormap с нормализацией относительно 1
    im = plt.imshow(result_array, cmap=colormap, aspect='auto', origin='lower', 
            interpolation='bilinear', extent=[0, float(d), 0, float(c)],
            norm=norm)
    plt.colorbar(im, label='Облученность нормализированная ')
    plt.xlabel('Y координата приемника (м)')
    plt.ylabel('X координата приемника (м)')
    plt.title('Поле облученности приемника')
    plt.grid(True, alpha=0.3)

    # Добавляем аннотации с min и max значениями
    plt.text(0.02, 0.98, f'Min: {min_irradiance:.3e} Вт/м²\nMax: {max_irradiance:.3e} Вт/м²',
            transform=plt.gca().transAxes, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10)

    plt.show()




asyncio.run(main(100))
    