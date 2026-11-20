from common import compare_groups

print('Гипотеза Г1')
print('Чем выше образование матери, тем выше итоговая оценка ученика')
print('Критерий подтверждения: средняя G3 растёт при каждом повышении Medu, а разница между уровнями 4 и 0 составляет не менее 1 балла')

table = compare_groups('Medu', name='g1_degtyareva')
means = table['средняя G3']
step_growth = means.diff().dropna().gt(0).all()
difference = means.loc[4] - means.loc[0]
confirmed = step_growth and difference >= 1.0

print('\nСредние оценки по уровню образования матери')
print(table.to_string())
print(f'Разница между уровнями 4 и 0: {difference:.2f} балла')
print('Рост средних оценок на каждом уровне:', 'да' if step_growth else 'нет')
print('Вывод по Г1:', 'подтверждена' if confirmed else 'опровергнута')
print('График сохранён в figures/g1_degtyareva.png')
