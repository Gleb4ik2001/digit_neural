import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from keras._tf_keras.keras.datasets import mnist
from tensorflow import keras
from keras._tf_keras.keras.layers import Dense, Flatten, Input
from tkinter import *
from PIL import Image, ImageDraw
import PIL

# Загрузка данных MNIST и обучение модели
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# Создание модели с использованием Input
model = keras.Sequential([
    Input(shape=(28, 28, 1)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)

# Оценка модели
model.evaluate(x_test, y_test_cat)

# GUI с использованием Tkinter и сеткой 28x28
class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Распознавание цифр по пикселям")

        self.canvas_size = 280  # Размер холста 280x280, 10 пикселей на один "пиксель" цифры
        self.pixel_size = self.canvas_size // 28  # Размер одного квадрата-пикселя

        self.canvas = Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg='white')
        self.canvas.pack()

        self.image = np.ones((28, 28)) * 255  # Изображение будет храниться как 28x28 массив (пустой белый фон)

        # Рисуем сетку 28x28
        self.draw_grid()

        # Кнопка "Определить"
        self.button_predict = Button(self.root, text="Определить", command=self.predict_digit)
        self.button_predict.pack()

        # Кнопка "Стереть"
        self.button_clear = Button(self.root, text="Стереть", command=self.clear_canvas)
        self.button_clear.pack()

        self.canvas.bind("<B1-Motion>", self.draw_digit)

    def draw_grid(self):
        """ Рисуем сетку 28x28 на холсте """
        for i in range(0, self.canvas_size, self.pixel_size):
            self.canvas.create_line([(i, 0), (i, self.canvas_size)], fill='gray', width=1)
            self.canvas.create_line([(0, i), (self.canvas_size, i)], fill='gray', width=1)

    def draw_digit(self, event):
        """ Рисование цифры в виде заполненных квадратов-пикселей """
        x, y = event.x, event.y
        x_pixel = x // self.pixel_size  # Переводим координаты мыши в координаты пикселя
        y_pixel = y // self.pixel_size

        if 0 <= x_pixel < 28 and 0 <= y_pixel < 28:  # Проверяем, что пиксель в пределах холста
            # Заполняем квадрат на холсте (черным)
            self.canvas.create_rectangle(x_pixel * self.pixel_size, y_pixel * self.pixel_size,
                                         (x_pixel + 1) * self.pixel_size, (y_pixel + 1) * self.pixel_size,
                                         fill='black')

            # Обновляем массив изображения, делаем соответствующий пиксель черным (0)
            self.image[y_pixel, x_pixel] = 0

    def clear_canvas(self):
        """ Очистка холста и массива изображения """
        self.canvas.delete("all")
        self.image = np.ones((28, 28)) * 255  # Сброс изображения (белый фон)
        self.draw_grid()  # Перерисовываем сетку

    def predict_digit(self):
        """ Предсказание нарисованной цифры с использованием модели """
        # Инвертируем изображение: цифра должна быть черной (0), фон белым (255)
        img_array = 255 - self.image  # Инвертируем цвета
        img_array = img_array / 255.0  # Нормализуем изображение
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)  # Добавляем канал

        res = model.predict(img_array)
        predicted_digit = np.argmax(res)
        print(f'Предсказанная цифра: {predicted_digit}')

        # Отображение изображения
        plt.imshow(np.squeeze(img_array), cmap=plt.cm.binary)
        plt.show()


# Создание окна Tkinter
root = Tk()
app = DigitRecognizerApp(root)
root.mainloop()
