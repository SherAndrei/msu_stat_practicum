# Задача 1
В файле "Shop.txt" представлены данные о доходах цветочного магазина в течение месяца работы. Можно ли утверждать на уровне значимости 3%, что средний суточный доход магазина больше 300 д.е.? Вернуть 1, если можно, и 0, если нельзя. Известно, что данные получены из нормального распределения.

# Задача 2
В файле "ABSamples.txt" приведены числовые значения двух признаков (A и B) некоторых объектов. Изучить вопрос о существовании монотонной зависимости между этими признаками. В качестве ответа вернуть значение p-value соответствующего теста, округленное с точностью до 7 знака после запятой. В случае нормального распределения данных использовать коэффициент корреляции Пирсона, в случае других распределений использовать коэффициент корреляции Кендалла.

# Задача 3
Были проведены клинические  испытания препара, предназначенного для понижения уровня инсулина в крови. Результаты проверки уровня инсулина пациентов до и после принятия препарата  представлены в файле "Insulin.txt". Можно ли утверждать, что препарат действительно работает? Вернуть в качестве ответа округленное до 7 знаков после запятой p-value параметрического или непараметрического теста (в зависимости от распределения). В качестве непараметрического теста использовать асимптотический критерий знаков. 

## Решение
    3.py

# Задача 4
В файле «SalaryData.txt» представлена зарплата учителей биологии в четырех городах страны N. Можно ли утверждать на уровне значимости 1%, что есть различия между средними зарплатами в 4 городах? В качестве ответа вернуть 1, если различия существуют, и p-value соответствующего теста (в зависимости от распределения данных), округленное с точностью до 7 знака после запятой, если различий нет.

## Решение
    4.py

# Задача 5
В файле "Boston.txt" представлены данные о домах в пригородах Бостона, в том числе такие характеристики как:
    medv – средняя стоимость домов
    rm – среднее количество комнат в доме
    crim – показатель криминальности района
    age – возраст домов

Исследуется зависимость стоимости домов от остальных перечисленных факторов. Интерес представляет стоимость дома при значениях факторов rm = 6.5, age = 70.0, crim = 0.02.

Проверить стандартизированные остатки регрессии на нормальность и проверить условие гомоскедастичности с помощью теста Уайта. Если хотя бы одно из данных двух условий не выполняется, вернуть (точечную) оценку стоимости дома при перечисленных значениях факторов. 
Если данные условия выполняются, вернуть в качестве ответа правый конец доверительного интервала для отклика с доверительной вероятностью 90%. 

Ответ округлить с точностью до 7 знака после запятой. 

Никакие дополнительные проверки модели (кроме нормальности остатков и условия гомоскедастичности) проводить не нужно.

Указание: если сохранить, например, в res_conf_int результат вычисления ДИ, то получить правый конец можно с помощью res_conf_int[0][1].

## Решение
    5.py

# Задача 6
В файле "BostonData.txt" представлены данные о домах в пригородах Бостона. Исследуется зависимость стоимости домов (medv) от остальных факторов путем построения модели гребневой регрессии (альфа=1, fit_intercept=True). 

Требуется:
1. Провести масштабирование данных.
2. Отобрать 5 признаков с самыми маленькими по модулю коэффициентами. Коэффициенты искать по train, делить выборку на train и test в отношении 75:25. 
3. Сравнить на test'е модель гребневой регрессии (альфа=1, fit_intercept=True) со всеми факторами с моделью гребневой регрессии (альфа=1, fit_intercept=True) со всеми факторами кроме отобранных на шаге 2. В качестве ответа вернуть модуль разности метрик RMSE этих моделей. Ответ округлить до 6 знака после запятой.

Указание: RMSE вычисляется как квадратный корень из MSE.

## Решение
    6.py
