import paramiko
import cv2
import os
import numpy as np
from PIL import Image

# Создаем объект SSHClient
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# Устанавливаем соединение
ssh.connect('192.168.2.74', username='danya', password='admin')

# Открываем SFTP-соединение
sftp = ssh.open_sftp()

# Получаем список файлов из удаленной папки
remote_dir = '/home/danya/dataSet/'
local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataSet')

# Обеспечим, что локальная папка существует
os.makedirs(local_dir, exist_ok=True)

# Копируем файлы с сервера на локальную машину
for filename in sftp.listdir(remote_dir):
    local_file_path = os.path.join(local_dir, filename)
    remote_file_path = remote_dir + filename
    sftp.get(remote_file_path, local_file_path)

# Закрываем SFTP-соединение и SSH
sftp.close()
ssh.close()

# Теперь продолжаем Ваш код для распознавания лиц...
# Создаем новый распознаватель лиц
recognizer = cv2.face.LBPHFaceRecognizer_create()
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# получаем картинки и подписи из датасета
def get_images_and_labels(datapath):
    image_paths = [os.path.join(datapath, f) for f in os.listdir(datapath)]
    images = []
    labels = []
    
    for image_path in image_paths:
        image_pil = Image.open(image_path).convert('L')
        image = np.array(image_pil, 'uint8')
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("face-", ""))
        faces = faceCascade.detectMultiScale(image)
        
        for (x, y, w, h) in faces:
            images.append(image[y:y + h, x:x + w])
            labels.append(nbr)
            cv2.imshow("Adding faces to training set...", image[y:y + h, x:x + w])
            cv2.waitKey(100)

    return images, labels

# получаем список картинок и подписей
dataPath = local_dir  # Используем локальный путь
images, labels = get_images_and_labels(dataPath)

# обучаем модель
recognizer.train(images, np.array(labels))

# сохраняем модель
trainer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trainer/trainer.yml')
recognizer.save(trainer_path)

# очищаем окна
cv2.destroyAllWindows()