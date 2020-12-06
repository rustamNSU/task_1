# курс "Параллельное программирование"

> Абдуллин Рустам, гр.19182 \
> email: <afaritovich@mail.ru>

## Task - 1
Решение СЛАУ реализованно с помощью метода BiCGStab.
Для распараллеливания метода был выделен в функцию оператор умножения матрицы на вектор,
и далее с помощью пакета MPI реализованна параллельная версия данной функции.
Для удобства были реализованны матрично-векторные операции **MatrixVector**.

В качестве тестовой матрицы использовалась *основа* (без постоянных физических коэффициентов)
матрицы влияния в методе граничных элементов (матрица является плотной и имеет диагональное преобладание).
Результаты выводятся в терминале, а также сохраняются в папке **output**.