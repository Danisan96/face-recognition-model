import paramiko
import cv2
import os
import numpy as np
from PIL import Image
import io

# Создаем объект SSHClient
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# Устанавливаем соединение
ssh.connect('192.168.2.74', username='danya', password='admin')

# Открываем SFTP-соединение
sftp = ssh.open_sftp()

# Получаем список файлов из удаленной папки
remote_dir = '/home/danya/dataSet/'

# Создаем распознаватель лиц
recognizer = cv2.face.LBPHFaceRecognizer_create()
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Получаем изображения и подписи из датасета
def get_images_and_labels():
    images = []
    labels = []    
    # Получаем список файлов из удаленной папки
    for filename in sftp.listdir(remote_dir):
        remote_file_path = remote_dir + filename
        # Загружаем изображение в память
        with sftp.open(remote_file_path) as file:
            img_data = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(img_data, cv2.IMREAD_GRAYSCALE)

            if image is not None:
                nbr = int(os.path.split(filename)[1].split(".")[0].replace("face-", ""))
                faces = faceCascade.detectMultiScale(image)

                for (x, y, w, h) in faces:
                    images.append(image[y:y+h, x:x+w])
                    labels.append(nbr)
                    cv2.imshow("Adding faces to training set...", image[y:y+h, x:x+w])
                    cv2.waitKey(100)

    return images, labels

# Предполагаем, что фотографии уже загружены на сервер и получаем список картинок и подписей
images, labels = get_images_and_labels()

# Обучаем модель
recognizer.train(images, np.array(labels))

# Сохраняем модель локально (можете убрать, если не нужно сохранять локально)
local_trainer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trainer/trainer.yml')
recognizer.save(local_trainer_path)

# Теперь загружаем модель на сервер
remote_trainer_path = '/home/danya/trainer/trainer.yml'
sftp.put(local_trainer_path, remote_trainer_path)

# Закрываем SFTP-соединение и SSH
sftp.close()
ssh.close()

# Очищаем окна
cv2.destroyAllWindows()