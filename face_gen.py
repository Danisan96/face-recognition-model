import cv2
import os
import pysftp

# Получаем путь к этому скрипту
path = os.path.dirname(os.path.abspath(__file__))

local_data_path = os.path.join(path, "dataSet")

detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Счетчик изображений
i = 0
# Расстояния от распознанного лица до рамки
offset = 50
# Запрашиваем номер пользователя
name = input('Введите номер пользователя: ')

# Получаем доступ к камере
video = cv2.VideoCapture(0)

# Подключение к SFTP-серверу
sftp_host = '192.168.2.74'
sftp_path = '/home/danya/dataSet/'
sftp_username = 'danya'
sftp_password = 'admin'

cnopts = pysftp.CnOpts()
cnopts.hostkeys = None  # Отключение проверки host key

with pysftp.Connection(sftp_host, username=sftp_username, password=sftp_password, cnopts=cnopts) as sftp:
    # Ваш код для работы с SFTP
    print("Соединено с SFTP-сервером")

    # Запускаем цикл
    while True:
        # Берем видеопоток
        ret, im = video.read()

        # Проверяем, что изображение не пустое
        if not ret or im is None:
            continue

        # Переводим всё в ч/б для простоты
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # Находим лица
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

        # Обрабатываем лица
        for (x, y, w, h) in faces:
            # Увеличиваем счетчик кадров
            i += 1
            # Формируем имя файла
            filename = f"face-{name}.{i}.jpg"
            local_path = os.path.join(path, "dataSet", filename)

            # Сохраняем локально
            cv2.imwrite(local_path, gray[y-offset:y+h+offset, x-offset:x+w+offset])

            # Загружаем файл на сервер
            sftp.put(local_path, sftp_path + filename)
            print(f"Загружено {filename} на сервер")

            # Формируем размеры окна для вывода лица
            cv2.rectangle(im, (x-50, y-50), (x+w+50, y+h+50), (225, 0, 0), 2)
            # Показываем очередной кадр
            cv2.imshow('im', im[y-offset:y+h+offset, x-offset:x+w+offset])
            # Делаем паузу
            cv2.waitKey(100)

        # Если у нас достаточно кадров
        if i > 100:
            # Освобождаем камеру
            video.release()
            # Удаляем все созданные окна
            cv2.destroyAllWindows()
            break
