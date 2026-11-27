
from pathlib import Path
import matplotlib.pyplot as plt

from common import (
    load_data,
    split,
    GlobalMean,
    ItemMean,
    Bias,
    SVD,
    compare_models
)


# Папка для сохранения графиков
ROOT = Path(__file__).parent.parent
FIGURES = ROOT / 'figures'
FIGURES.mkdir(exist_ok=True)


# ==================================================
# 1. Загрузка данных
# ==================================================

df = load_data()

print('ОБЩИЙ АНАЛИЗ ДАННЫХ')
print('=' * 50)

print(f'Всего оценок: {len(df)}')
print(f'Пользователей: {df["user_id"].nunique()}')
print(f'Товаров: {df["item_id"].nunique()}')

print('\n')


# ==================================================
# 2. ТОП-10 самых популярных товаров
# ==================================================

items = (
    df.groupby('item_id')
    .agg(
        количество_оценок=('rating', 'count'),
        средняя_оценка=('rating', 'mean'),
        категория=('category', 'first'),
        бренд=('brand', 'first')
    )
    .sort_values('количество_оценок', ascending=False)
)

print('ТОП-10 САМЫХ ПОПУЛЯРНЫХ ТОВАРОВ')
print(items.head(10).round(3).to_string())


# ==================================================
# 3. ТОП-10 самых высоко оценённых товаров
#    У каждого товара должно быть минимум 50 оценок
# ==================================================

best_items = (
    items[items['количество_оценок'] >= 50]
    .sort_values('средняя_оценка', ascending=False)
)

print('\nТОП-10 САМЫХ ВЫСОКО ОЦЕНЁННЫХ ТОВАРОВ')
print(best_items.head(10).round(3).to_string())


# ==================================================
# 4. Распределение оценок от 1 до 5
# ==================================================

rating_counts = df['rating'].value_counts().sort_index()
rating_percent = df['rating'].value_counts(normalize=True).sort_index() * 100

rating_table = (
    rating_counts.to_frame('Количество')
    .join(rating_percent.to_frame('Процент'))
)

print('\nРАСПРЕДЕЛЕНИЕ ОЦЕНОК')
print(rating_table.round(2).to_string())

plt.figure(figsize=(8, 5))
plt.bar(rating_counts.index.astype(str), rating_counts.values)
plt.title('Распределение оценок покупателей')
plt.xlabel('Оценка')
plt.ylabel('Количество оценок')
plt.tight_layout()
plt.savefig(FIGURES / 'zhanna_rating_distribution.png', dpi=150)
plt.close()


# ==================================================
# 5. Количество покупок у пользователей
# ==================================================

user_purchases = df.groupby('user_id').size()

print('\nКОЛИЧЕСТВО ПОКУПОК У ПОЛЬЗОВАТЕЛЕЙ')
print(user_purchases.describe().round(2).to_string())

print('\nСколько пользователей совершили:')
print(f'1 покупку: {(user_purchases == 1).sum()}')
print(f'2 покупки: {(user_purchases == 2).sum()}')
print(f'3 и более покупок: {(user_purchases >= 3).sum()}')

plt.figure(figsize=(8, 5))
plt.hist(user_purchases, bins=30)
plt.title('Распределение количества покупок у пользователей')
plt.xlabel('Количество покупок')
plt.ylabel('Количество пользователей')
plt.tight_layout()
plt.savefig(FIGURES / 'zhanna_user_purchases.png', dpi=150)
plt.close()


# ==================================================
# 6. ГИПОТЕЗА Г10
# Нужен ли сложный алгоритм?
# Способ В — сравнение точности алгоритмов
# ==================================================

print('\n')
print('=' * 50)
print('ГИПОТЕЗА Г10')
print('Сравнение точности рекомендательных алгоритмов')
print('=' * 50)

train, test = split(df)

print(f'Обучающая выборка: {len(train)}')
print(f'Проверочная выборка: {len(test)}')

results = compare_models(
    [
        GlobalMean(),
        ItemMean(),
        Bias(),
        SVD()
    ],
    train,
    test,
    name='zhanna_g10'
)

print('\nРЕЗУЛЬТАТЫ СРАВНЕНИЯ АЛГОРИТМОВ')
print(results.to_string(index=False))

from common import evaluate

print('\nТОЧНЫЕ РЕЗУЛЬТАТЫ SVD И BIAS:')

for model in [Bias(), SVD()]:
    model.fit(train)
    metrics = evaluate(model, test)
    print(
        model.name,
        'MAE:', f'{metrics["mae"]:.6f}',
        'RMSE:', f'{metrics["rmse"]:.6f}'
    )


print('\nГрафики сохранены в папке figures.')
