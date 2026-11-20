"""
Сергей П., гипотеза Г3: кто раньше заваливал предметы, получает заметно ниже.

Способ А: сравниваем среднюю G3 по группам признака failures
(число прошлых завалов: 0, 1, 2, 3+).

Правило вывода (записано ДО проверки):
    подтверждена, если средняя G3 у группы с failures >= 1 ниже,
    чем у группы с failures = 0, минимум на 1 балл.
"""

from common import compare_groups

table = compare_groups('failures', name='g3_failures')
print('Способ А — средняя G3 по числу прошлых завалов (failures):')
print(table)

mean0 = table.loc[0, 'средняя G3']
mean_bad = table.loc[table.index > 0, 'средняя G3'].min()
diff = mean0 - mean_bad
print(f'\nСредняя G3 при failures=0: {mean0:.2f}')
print(f'Минимальная средняя G3 при failures>=1: {mean_bad:.2f}')
print(f'Разница: {diff:.2f} балла ->',
      'подтверждена' if diff >= 1 else 'опровергнута')
print('\nКартинка: figures/g3_failures.png')