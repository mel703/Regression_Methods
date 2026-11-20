"""
Дарья, гипотеза Г6: свободное время и кружки модели не нужны.

Признаки: freetime (свободное время, 1-5) и activities (кружки, да/нет).
Способ Б: обучаем модель со всеми признаками и без этих двух, сравниваем MAE.

Правило вывода (записано ДО проверки):
    подтверждена, если без этих признаков ошибка выросла меньше чем на 0,05 балла
    (то есть модель справляется так же хорошо, значит признаки не нужны).
"""

from common import compare_without

THRESHOLD = 0.05  # баллов

# ---------- Способ Б: убрать признаки ----------
# Гипотеза: «свободное время и кружки модели не нужны».
res = compare_without(['freetime', 'activities'], name='g6_freetime_activities')

print('Способ Б — ошибка со всеми признаками: {:.3f} балла'.format(res["mae_all"]))
print('Ошибка без признаков (freetime, activities): {:.3f} балла'.format(res["mae_without"]))
print('Разница: {:+.3f} балла'.format(res["difference"]))

print('Г6:', 'подтверждена' if res['difference'] < THRESHOLD else 'опровергнута')

print('\nКартинка: figures/g6_freetime_activities.png')