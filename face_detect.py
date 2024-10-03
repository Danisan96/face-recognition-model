import cv2
import os
import paramiko
import numpy as np

# Настройки SFTP
hostname = "192.168.2.74"
port = 22
username = "danya"
password = "admin"
remote_path = "/home/danya/trainer/trainer.yml"
local_path = "trainer.yml"

# Функция для загрузки trainer.yml с SFTP
def download_trainer_file():
    try:
        # Создаем SSH-клиент
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        # Подключаемся к серверу
        ssh.connect(hostname, port, username, password)
        # Создаем SFTP-сессию
        sftp = ssh.open_sftp()
        # Загружаем файл
        sftp.get(remote_path, local_path)
        sftp.close()
        ssh.close()
        print("Файл успешно загружен с SFTP.")
    except Exception as e:
        print("Ошибка загрузки файла:", e)

# Загружаем trainer.yml
download_trainer_file()

# создаём новый распознаватель лиц
recognizer = cv2.face.LBPHFaceRecognizer_create()
# добавляем в него модель, которую мы обучили на прошлом этапе
recognizer.read(local_path)
# указываем, что мы будем искать лица по примитивам Хаара
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# получаем доступ к камере
cam = cv2.VideoCapture(0)
# настраиваем шрифт для вывода подписей
font = cv2.FONT_HERSHEY_SIMPLEX

# запускаем цикл
while True:
    # получаем видеопоток
    ret, im = cam.read()
    # переводим его в ч/б
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # определяем лица на видео
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    # перебираем все найденные лица
    for (x, y, w, h) in faces:
        # получаем id пользователя
        nbr_predicted, coord = recognizer.predict(gray[y:y+h, x:x+w])
        # рисуем прямоугольник вокруг лица
        cv2.rectangle(im, (x-50, y-50), (x+w+50, y+h+50), (225, 0, 0), 2)

        # если мы знаем id пользователя
        if nbr_predicted == 7:
            nbr_predicted = 'Danya Golubenko'
        if nbr_predicted == 7:
            nbr_predicted = 'Vova'
        
        # добавляем текст к рамке
        cv2.putText(im, str(nbr_predicted), (x, y+h), font, 1.1, (0, 255, 0))
    
    # выводим окно с изображением с камеры
    cv2.imshow('Face recognition', im)

    # выход при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# освобождаем ресурсы
cam.release()
cv2.destroyAllWindows()