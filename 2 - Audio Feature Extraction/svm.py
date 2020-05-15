import numpy as np
import os
import pickle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from spectrogram import get_spectrogram, get_noise_augmented_spectrogram
from mfcc import get_mfcc, get_noise_augmented_mfcc

# feature = "mfcc"
feature = "spectrogram"
noise_augmentation = False

training_path = os.path.join('../data/training')
validation_path = os.path.join('../data/validation')
noise_path = os.path.join('../data/_background_noise_')

classes_dict = {'zero' : 0, 'one' : 1, 'two' : 2, 'three' : 3, 'four' : 4, 'five' : 5, 'six' : 6, 'seven' : 7, 'eight' : 8, 'nine' : 9}
classes_name = classes_dict.keys()

X_train = []
y_train = []

if os.path.exists('../pickles/' + feature + '.nine.train.pkl'):
    with open('../pickles/' + feature + '.nine.train.pkl', 'rb') as f:
        X_train = pickle.load(f)
        f.close()

if len(X_train) == 0:
    for class_name in classes_name:
        count = 0
        class_folder_path = os.path.join(training_path, class_name)
        audios = os.listdir(class_folder_path)
        for audio in audios:
            if noise_augmentation:
                noises = os.listdir(noise_path)
                for noise in noises:
                    if feature == "mfcc":
                        audio_mfcc = get_noise_augmented_mfcc(os.path.join(class_folder_path, audio), os.path.join(noise_path, noise))
                        X_train.append(audio_mfcc.flatten())
                    else:
                        audio_spectrogram = get_noise_augmented_spectrogram(os.path.join(class_folder_path, audio), os.path.join(noise_path, noise))
                        X_train.append(audio_spectrogram.flatten())
                    y_train.append(classes_dict[class_name])
            else:
                if feature == "mfcc":
                    audio_mfcc = get_mfcc(os.path.join(class_folder_path, audio))
                    X_train.append(audio_mfcc.flatten())
                else:
                    audio_spectrogram = get_spectrogram(os.path.join(class_folder_path, audio))
                    X_train.append(audio_spectrogram.flatten())
                y_train.append(classes_dict[class_name])
            count += 1
            print(count, "done from class", class_name)
        with open("../pickles/" + feature + '.' + class_name + '.train.pkl', 'wb') as f:
            pickle.dump(X_train, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        print("pickled", class_name)
else:
    print("loaded pickled X_train with length", len(X_train), "and shape", X_train[0].shape)
    for class_name in classes_name:
        count = 0
        class_folder_path = os.path.join(training_path, class_name)
        audios = os.listdir(class_folder_path)
        for audio in audios:
            y_train.append(classes_dict[class_name])
            count += 1
            if count % 100 == 0:
                print(count, "done from class", class_name)

X_test = []
y_test = []

if os.path.exists('../pickles/' + feature + '.nine.test.pkl'):
    with open('../pickles/' + feature + '.nine.test.pkl', 'rb') as f:
        X_test = pickle.load(f)
        f.close()

if len(X_test) == 0:
    for class_name in classes_name:
        count = 0
        class_folder_path = os.path.join(validation_path, class_name)
        audios = os.listdir(class_folder_path)
        for audio in audios:
            if feature == "mfcc":
                audio_mfcc = get_mfcc(os.path.join(class_folder_path, audio))
                X_test.append(audio_mfcc.flatten())
            else:
                audio_spectrogram = get_spectrogram(os.path.join(class_folder_path, audio))
                X_test.append(audio_spectrogram.flatten())
            y_test.append(classes_dict[class_name])
            count += 1
            print(count, "done from class", class_name)
        with open("../pickles/" + feature + '.' + class_name + '.test.pkl', 'wb') as f:
            pickle.dump(X_test, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        print("pickled", class_name)
else:
    print("loaded pickled X_test with length", len(X_test), "and shape", X_test[0].shape)
    for class_name in classes_name:
        count = 0
        class_folder_path = os.path.join(validation_path, class_name)
        audios = os.listdir(class_folder_path)
        for audio in audios:
            y_test.append(classes_dict[class_name])
            count += 1
            if count % 100 == 0:
                print(count, "done from class", class_name)

X_train = np.log1p(np.array(X_train))
X_test = np.log1p(np.array(X_test))

clf = SVC(verbose=True)
clf.fit(X_train, y_train)
with open('../pickles/' + feature + '.classifier.pkl', 'wb') as f:
    pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)
    f.close()
print("pickled classifier")
y_predict = clf.predict(X_test)
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))