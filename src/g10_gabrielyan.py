"""
ипотеза Г10: 15 учеников с итоговой оценкой 0 мешают модели.

Почему так думаем: у всех 15 ноль пропусков и почти у всех слабые оценки за год —
похоже, они бросили школу. В анкете этого нет, поэтому модель не может их угадать.

Как проверяем (способ Б, только убираем не столбец, а строки):
    модель 1 — учится на всех учениках обучающей части;
    модель 2 — учится без учеников с нулём;
    обе проверяем на ОДНОЙ И ТОЙ ЖЕ тестовой части (195 учеников), чтобы сравнение было честным.
Правило вывода (записано до проверки):
    подтверждена, если без нулей средняя ошибка меньше хотя бы на 0,05 балла.
"""

from matplotlib import pyplot as plt

from common import FIGURES_PATH, evaluate, get_model, load_data, split

THRESHOLD = 0.05

X, y = load_data()
X_train, X_test, y_train, y_test = split(X, y)
zero_train = y_train == 0
print(f'Учеников с итоговым нулём: обучение {zero_train.sum()}, проверка {(y_test == 0).sum()}')

model_all = get_model().fit(X_train, y_train)
model_cut = get_model().fit(X_train[~zero_train], y_train[~zero_train])

res_all = evaluate(model_all, X_test, y_test)
res_cut = evaluate(model_cut, X_test, y_test)
gain = res_all['mae'] - res_cut['mae']

print(f'Модель 1 (все ученики):  ошибка {res_all["mae"]:.3f} балла, R² {res_all["r2"]:.3f}')
print(f'Модель 2 (без нулей):    ошибка {res_cut["mae"]:.3f} балла, R² {res_cut["r2"]:.3f}')
print(f'Ошибка уменьшилась на {gain:.3f} балла')
print('Г10:', 'подтверждена' if gain > THRESHOLD else 'опровергнута')

# Дополнительно: та же проверка только на обычных учениках (без нулей в тестовой части)
normal = y_test > 0
mae_all_n = evaluate(model_all, X_test[normal], y_test[normal])['mae']
mae_cut_n = evaluate(model_cut, X_test[normal], y_test[normal])['mae']
print(f'Только обычные ученики: ошибка {mae_all_n:.3f} -> {mae_cut_n:.3f} балла')

# Картинка для слайда
FIGURES_PATH.mkdir(exist_ok=True)
values = [res_all['mae'], res_cut['mae']]
ax = plt.gca()
ax.bar(['учится на всех', 'учится без 15 нулей'], values, color=['#2B5CA8', '#9CB3D9'])
ax.set_ylabel('Средняя ошибка, баллов (MAE)')
ax.set_title('Г10: ошибка модели на одной и той же тестовой части')
for i, v in enumerate(values):
    ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig(FIGURES_PATH / 'g10_gabrielyan.png', dpi=150)
plt.close()
print('Картинка: figures/g10_gabrielyan.png')
