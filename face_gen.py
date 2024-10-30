import cv2
import os
import pysftp
import logging
import numpy as np
import tempfile

logging.basicConfig(filename='face_capture.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Получаем путь к этому скрипту
path = os.path.dirname(os.path.abspath(__file__))

detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Счетчик изображений
i = 0
# Расстояния от распознанного лица до рамки
offset = 50
# Запрашиваем номер пользователя
name = input('Введите номер пользователя: ')

# Подключаемся к IP-камере
ip_camera_url = "rtsp://admin:admin123456@192.168.1.17:8554/profile0"  # замените на URL Вашей камеры
video = cv2.VideoCapture(ip_camera_url)

sftp_host = '192.168.2.74'
sftp_path = '/home/danya/dataSet/'
sftp_username = 'danya'
sftp_password = 'admin'

cnopts = pysftp.CnOpts()
cnopts.hostkeys = None  # Отключение проверки host key

with pysftp.Connection(sftp_host, username=sftp_username, password=sftp_password, cnopts=cnopts) as sftp:
    print("Соединено с SFTP-сервером")

    # Запускаем цикл
    while True:
        # Берем видеопоток
        ret, im = video.read()

        # Проверяем, что изображение не пустое
        if not ret or im is None:
            logging.warning("Не удалось получить кадр с камеры.")
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

            # Берем область с лицом и добавляем отступы
            face_img = gray[y-offset:y+h+offset, x-offset:x+w+offset]

            # Преобразуем изображение в формат, который можно отправить на SFTP
            _, buffer = cv2.imencode('.jpg', face_img)

            # Создаем временный файл для загрузки
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_file.write(buffer)
                temp_file_path = temp_file.name

            # Загружаем файл на сервер
            sftp.put(temp_file_path, sftp_path + filename)
            print(f"Загружено {filename} на сервер")

            # Формируем размеры окна для вывода лица
            cv2.rectangle(im, (x-50, y-50), (x+w+50, y+h+50), (225, 0, 0), 2)
            # Показываем очередной кадр
            cv2.imshow('im', im[y-offset:y+h+offset, x-offset:x+w+offset])
            # Делаем паузу
            cv2.waitKey(100)

        # Если у нас достаточно кадров
        if i > 300:
            # Освобождаем камеру
            video.release()
            # Удаляем все созданные окна
            cv2.destroyAllWindows()
            break
