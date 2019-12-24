# Bengali_Vehicle_Classifier
Vehicle detection using Deep Learning and Computer Vission
#DATA TRAIN WITH GOOGLE COLAB

import json
from pprint import pprint
from google.colab import auth
from googleapiclient.discovery import build
import io
from googleapiclient.http import MediaIoBaseDownload
import pickle

auth.authenticate_user()
drive_service = build('drive', 'v3')

def _create_file_request(file_id):
    return drive_service.files().get_media(fileId=file_id)
 
 
def _download_response_bytes(request, print_progress=False):
    downloaded = io.BytesIO()
    downloader = MediaIoBaseDownload(downloaded, request)
   
    for status in _progbar(downloader):
        if print_progress:
            print("Downloaded {}/{} bytes".format(status.resumable_progress, status.total_size))
 
    downloaded.seek(0)
    return downloaded.read()
 
 
def _progbar(downloader):
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        yield status
 
 
def get_file_id(name):
    return get_matching_files(name)[0]['id']
 
 
def move_from_drive_to_disk(file_names, file_destinations):
    for file_name, dest in zip(file_names, file_destinations):
        file_id = get_file_id(file_name)
        print('Downloading file: "{}"'.format(file_name))
        file_bytes = _download_response_bytes(_create_file_request(file_id), print_progress=True)
        with open(dest, "wb") as f:
            f.write(file_bytes)
 
 
def load_pickled_files(file_names):
    for name in file_names:
        yield pickle.load(open(name, "rb"))
       
 
def get_matching_files(name):
    drive_files_response = _download_response_bytes(drive_service.files().list())
    drive_files_response_dict = json.loads(
        drive_files_response.decode('utf-8')
    )
    drive_files_dict = drive_files_response_dict['files']
 
    matching_records = [
        record
        for record in drive_files_dict
        if record['name'] == name
    ]
 
    no_records = len(matching_records)
    if no_records == 0:
        raise ValueError('no such file: "{}" on your Google Drive'.format(name))
    elif no_records > 1:
        print('warning: multiple matches for file "{}"'.format(name))
    return matching_records
 
 
example_file_name = ['FILE_NAME_x.npy', 'FILE_NAME_y.npy']
for i in example_file_name:
    example_file_id = get_file_id(i)
    move_from_drive_to_disk([i], [i])
    
 
import numpy as np
x= np.load('cnn_cat_x.npy')
y= np.load('cnn_cat_y.npy')
y=np.transpose(y)
x = np.expand_dims(x, axis=3)
print(x.shape)
print(y.shape)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Input, Conv2D, Flatten, Activation

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=0)


classifier= Sequential()

classifier.add(Conv2D(10,(5,5), input_shape = (28,28,1), activation ='relu'))
classifier.add(Conv2D(15,(5,5), activation = 'relu'))
classifier.add(Conv2D(20,(5,5), activation = 'relu'))

classifier.add(Flatten())
classifier.add(Dense(64, activation ='relu'))

classifier.add(Dense(1, activation ='sigmoid'))

classifier.compile(optimizer ='Adam', loss = 'binary_crossentropy', metrics =['accuracy'])


classifier.fit(X_train, y_train, validation_data= (X_test, y_test), epochs=20)

import os
classifier.save_weights('Weight_95.h5')
os.listdir()

from google.colab import files
files.download('Weight_95.h5')
