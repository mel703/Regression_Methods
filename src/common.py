"""
Общий файл ЛР 3 «Успеваемость» (common.py). Пишет и меняет только Мелине.

Задача: по анкете ученика предсказать итоговую оценку G3 (0–20).
Данные: data/student-por.csv — 649 учеников, португальский язык.

Все скрипты команды начинаются со строки:
    from common import load_data, load_table, split, get_model, evaluate, compare_groups, compare_without

Что здесь есть:
    load_table()       — исходная таблица как есть (слова не переведены) — для способа А
    load_data()        — признаки X в числах и оценка y = G3 — для модели
    split()            — деление 70/30, одинаковое у всей команды
    get_model()        — одна модель на всех: случайный лес, 300 деревьев
    evaluate()         — ошибка модели: MAE (в баллах), RMSE, R²
    compare_groups()   — способ А: средняя G3 по группам признака + картинка
    compare_without()  — способ Б: ошибка модели со всеми признаками и без указанных + картинка
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / 'data' / 'student-por.csv'
FIGURES_PATH = ROOT / 'figures'

TARGET = 'G3'            # что предсказываем
DROPPED = ['G1', 'G2']   # оценки за 1-й и 2-й период: убираем, предсказываем только по анкете

# Признаки с двумя вариантами: какой вариант становится 1 (второй — 0)
BINARY = {
    'school': 'GP',      # 1 = Gabriel Pereira, 0 = Mousinho da Silveira
    'sex': 'M',          # 1 = мужской, 0 = женский
    'address': 'U',      # 1 = город, 0 = деревня
    'famsize': 'GT3',    # 1 = больше 3 человек в семье
    'Pstatus': 'T',      # 1 = родители живут вместе
    'schoolsup': 'yes',  # остальные: 1 = да, 0 = нет
    'famsup': 'yes',
    'paid': 'yes',
    'activities': 'yes',
    'nursery': 'yes',
    'higher': 'yes',
    'internet': 'yes',
    'romantic': 'yes',
}

# Признаки с несколькими вариантами словами: каждый вариант — отдельный столбец да/нет
# (например, Mjob -> Mjob_teacher, Mjob_health, Mjob_services, Mjob_at_home, Mjob_other)
NOMINAL = ['Mjob', 'Fjob', 'reason', 'guardian']

# Остальные 13 признаков уже числа и остаются как есть:
# age, Medu, Fedu, traveltime, studytime, failures, famrel, freetime, goout, Dalc, Walc, health, absences


def load_table(path=DATA_PATH):
    """Исходная таблица целиком, слова не переведены. Для способа А и для описания признаков."""
    df = pd.read_csv(path, sep=';')
    if df.shape != (649, 33):
        raise ValueError(f'Ожидается таблица 649 x 33, получено {df.shape}. Тот ли файл?')
    if df.isna().any().any():
        raise ValueError('В данных есть пропуски.')
    return df


def load_data(path=DATA_PATH):
    """
    Признаки X (все в числах) и оценка y = G3.
    G1 и G2 убраны. Пропусков в данных нет, все 649 учеников остаются.
    """
    df = load_table(path)
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET] + DROPPED)

    for col, one in BINARY.items():
        X[col] = (X[col] == one).astype(int)
    X = pd.get_dummies(X, columns=NOMINAL, dtype=int)
    return X, y


def split(X, y):
    """70 % учеников — обучение, 30 % — проверка. Одинаково у всей команды."""
    return train_test_split(X, y, test_size=0.3, random_state=42)


def get_model(random_state=42):
    """Одна модель на всех. Настройки для отчёта: случайный лес, 300 деревьев, random_state = 42."""
    return RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)


def evaluate(model, X_test, y_test):
    """
    Ошибка уже обученной модели на тестовой части:
        mae  — на сколько баллов в среднем ошибается (главное число)
        rmse — то же, но сильнее наказывает большие промахи
        r2   — доля различий в оценках, которую объясняет модель (1 — всё, 0 — ничего)
    """
    pred = model.predict(X_test)
    return {
        'mae': mean_absolute_error(y_test, pred),
        'rmse': float(np.sqrt(mean_squared_error(y_test, pred))),
        'r2': r2_score(y_test, pred),
    }


def baseline(y_train, y_test):
    """Точка отсчёта: «всем ставим среднюю оценку». Модель должна ошибаться меньше."""
    pred = np.full(len(y_test), y_train.mean())
    return mean_absolute_error(y_test, pred)


def _columns_of(X, features):
    """Столбцы X, которые относятся к признакам (у Mjob это все Mjob_...)."""
    cols = []
    for f in features:
        found = [c for c in X.columns if c == f or c.startswith(f + '_')]
        if not found:
            raise ValueError(f'Признака {f} нет в данных.')
        cols += found
    return cols


def compare_groups(column, name=None):
    """
    Способ А: средняя итоговая оценка G3 по группам признака (по всем 649 ученикам).
    Возвращает таблицу: значение признака, сколько учеников, средняя G3.
    name — имя картинки в figures/ (например 'g5_walc'); без name картинка не сохраняется.
    """
    df = load_table()
    table = (df.groupby(column)[TARGET]
               .agg(['count', 'mean'])
               .rename(columns={'count': 'учеников', 'mean': 'средняя G3'})
               .round(2))
    if name:
        FIGURES_PATH.mkdir(exist_ok=True)
        ax = table['средняя G3'].plot.bar(color='#2B5CA8', rot=0)
        ax.set_xlabel(column)
        ax.set_ylabel('Средняя итоговая оценка G3')
        ax.set_title(f'Средняя G3 по группам: {column}')
        for i, v in enumerate(table['средняя G3']):
            ax.text(i, v, f'{v:.1f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(FIGURES_PATH / f'{name}.png', dpi=150)
        plt.close()
    return table


def compare_without(features, name=None):
    """
    Способ Б: обучаем одну и ту же модель со всеми признаками и без указанных.
    features — список признаков, например ['internet'] или ['Mjob', 'Fjob'].
    Возвращает {'mae_all': ..., 'mae_without': ..., 'difference': ...} (в баллах).
    difference > 0 — без признаков модель ошибается больше, то есть они нужны.
    """
    X, y = load_data()
    X_train, X_test, y_train, y_test = split(X, y)
    cols = _columns_of(X, features)

    full = get_model().fit(X_train, y_train)
    cut = get_model().fit(X_train.drop(columns=cols), y_train)
    mae_all = evaluate(full, X_test, y_test)['mae']
    mae_without = evaluate(cut, X_test.drop(columns=cols), y_test)['mae']

    if name:
        FIGURES_PATH.mkdir(exist_ok=True)
        labels = ['все признаки', 'без ' + ', '.join(features)]
        ax = plt.gca()
        ax.bar(labels, [mae_all, mae_without], color=['#2B5CA8', '#9CB3D9'])
        ax.set_ylabel('Средняя ошибка, баллов (MAE)')
        ax.set_title('Ошибка модели с признаком и без')
        for i, v in enumerate([mae_all, mae_without]):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(FIGURES_PATH / f'{name}.png', dpi=150)
        plt.close()

    return {'mae_all': mae_all, 'mae_without': mae_without, 'difference': mae_without - mae_all}


if __name__ == '__main__':
    # Быстрая проверка, что всё работает
    X, y = load_data()
    print('Признаков после перевода в числа:', X.shape[1], '| учеников:', len(X))
    X_train, X_test, y_train, y_test = split(X, y)
    print('Обучение:', len(X_train), '| проверка:', len(X_test))

    model = get_model().fit(X_train, y_train)
    res = evaluate(model, X_test, y_test)
    print(f'Модель: ошибка {res["mae"]:.2f} балла, RMSE {res["rmse"]:.2f}, R² {res["r2"]:.3f}')
    print(f'«Всем среднюю оценку»: ошибка {baseline(y_train, y_test):.2f} балла')
