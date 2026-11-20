"""
Сергей П., гипотеза Г4: кто хочет в вуз (higher = yes), учится лучше.

Способ А: сравниваем среднюю G3 по группам признака higher.

Правило вывода (записано ДО проверки):
    подтверждена, если средняя G3 у группы higher = yes выше,
    чем у группы higher = no, минимум на 1 балл.
"""

from common import compare_groups

table = compare_groups('higher', name='g4_higher')
print('Способ А — средняя G3 по желанию поступать в вуз (higher):')
print(table)

mean_yes = table.loc['yes', 'средняя G3']
mean_no = table.loc['no', 'средняя G3']
diff = mean_yes - mean_no
print(f'\nСредняя G3 при higher=yes: {mean_yes:.2f}')
print(f'Средняя G3 при higher=no:  {mean_no:.2f}')
print(f'Разница: {diff:.2f} балла ->',
      'подтверждена' if diff >= 1 else 'опровергнута')
print('\nКартинка: figures/g4_higher.png')