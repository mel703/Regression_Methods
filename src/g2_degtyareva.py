from common import compare_without

threshold = 0.05

print('Гипотеза Г2')
print('Профессии родителей не нужны модели для прогнозирования итоговой оценки')
print(f'Гипотеза подтверждается, если ошибка без профессий вырастет менее чем на {threshold:.2f} балла')

result = compare_without(['Mjob', 'Fjob'], name='g2_degtyareva')

print(f'Ошибка со всеми признаками: {result["mae_all"]:.3f} балла')
print(f'Ошибка без профессий родителей: {result["mae_without"]:.3f} балла')
print(f'Изменение ошибки: {result["difference"]:+.3f} балла')
print('Вывод по Г2:', 'подтверждена' if result['difference'] < threshold else 'опровергнута')
print('График сохранён в figures/g2_degtyareva.png')
