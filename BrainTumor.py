import os
import pickle
import ftplib
import numpy as np
import matplotlib.pyplot as plt
import cv2
cv = cv2  # Alias for cv2 compatibility
import imutils
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics

# Suppress verbose TensorFlow / Keras warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Compatible Keras imports (supports tf-keras, tensorflow.keras, or standalone keras)
try:
    import tf_keras as keras
    from tf_keras.utils import to_categorical
    from tf_keras.layers import MaxPooling2D, Dense, Dropout, Activation, Flatten, Conv2D, Conv2D as Convolution2D
    from tf_keras.models import Sequential, model_from_json
except ImportError:
    try:
        import tensorflow as tf
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.layers import MaxPooling2D, Dense, Dropout, Activation, Flatten, Conv2D, Conv2D as Convolution2D
        from tensorflow.keras.models import Sequential, model_from_json
    except ImportError:
        import keras
        try:
            from keras.utils import to_categorical
        except ImportError:
            from keras.utils.np_utils import to_categorical
        from keras.layers import MaxPooling2D, Dense, Dropout, Activation, Flatten, Conv2D, Conv2D as Convolution2D
        from keras.models import Sequential, model_from_json

import tkinter
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import filedialog
from tkinter import ttk
from tkinter import Text, Scrollbar, Button, Label, END
from tkinter.filedialog import askopenfilename

main = tkinter.Tk()
main.title("Identifying Brain Tumor using X-Ray Images")  # designing main screen
main.geometry("1300x720")

global filename
global accuracy
X = []
Y = []
global classifier
classifier = None
disease = ['No Tumor Detected', 'Tumor Detected']

# Load segmentation model
with open('Model/segmented_model.json', "r") as json_file:
    loaded_model_json = json_file.read()
    segmented_model = model_from_json(loaded_model_json)

segmented_model.load_weights("Model/segmented_weights.h5")
if hasattr(segmented_model, '_make_predict_function'):
    segmented_model._make_predict_function()
elif hasattr(segmented_model, 'make_predict_function'):
    segmented_model.make_predict_function()

def edgeDetection():
    img = cv2.imread('myimg.png')
    orig = cv2.imread('test1.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    min_area = 0.95 * 180 * 35
    max_area = 1.05 * 180 * 35
    result = orig.copy()
    for c in contours:
        area = cv2.contourArea(c)
        cv2.drawContours(result, [c], -1, (0, 0, 255), 10)
        if area > min_area and area < max_area:
            cv2.drawContours(result, [c], -1, (0, 255, 255), 10)
    return result    

def tumorSegmentation(img_path):
    global segmented_model
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    img = img.reshape(1, 64, 64, 1)
    img = (img - 127.0) / 127.0
    preds = segmented_model.predict(img)
    preds = preds[0]
    print("Segmentation shape:", preds.shape)
    orig = cv2.imread(img_path, 0)
    orig = cv2.resize(orig, (300, 300), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("test1.png", orig)    
    segmented_image = cv2.resize(preds, (300, 300), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("myimg.png", segmented_image * 255)
    edge_detection = edgeDetection()
    return segmented_image * 255, edge_detection

def uploadDataset():  # function to upload dataset
    global filename
    selected_dir = filedialog.askdirectory(initialdir=".")
    if not selected_dir:
        return
    filename = selected_dir
    text.delete('1.0', END)
    text.insert(END, filename + " loaded\n")

def datasetPreprocessing():
    global X
    global Y
    X = []
    Y = []
    if os.path.exists('Model/myimg_data.txt.npy'):
        X = np.load('Model/myimg_data.txt.npy')
        Y = np.load('Model/myimg_label.txt.npy')
    else:
        if 'filename' not in globals() or not filename:
            messagebox.showinfo("Error", "Please upload dataset first!")
            return
        for root, dirs, directory in os.walk(filename + "/no"):
            for i in range(len(directory)):
                name = directory[i]
                img = cv2.imread(filename + "/no/" + name, 0)  # reading images
                ret2, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  # processing and normalization images
                img = cv2.resize(img, (128, 128))  # resizing images
                im2arr = np.array(img)  # extract features from images
                im2arr = im2arr.reshape(128, 128, 1)
                X.append(im2arr)
                Y.append(0)
                print(filename + "/no/" + name)

        for root, dirs, directory in os.walk(filename + "/yes"):
            for i in range(len(directory)):
                name = directory[i]
                img = cv2.imread(filename + "/yes/" + name, 0)
                ret2, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                img = cv2.resize(img, (128, 128))
                im2arr = np.array(img)
                im2arr = im2arr.reshape(128, 128, 1)
                X.append(im2arr)
                Y.append(1)
                print(filename + "/yes/" + name)
                
        X = np.asarray(X)
        Y = np.asarray(Y)            
        np.save("Model/myimg_data.txt", X)
        np.save("Model/myimg_label.txt", Y)
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    cv2.imshow('Sample Image (Press any key to close)', X[20])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    text.insert(END, "Total number of images found in dataset : " + str(len(X)) + "\n")
    text.insert(END, "Total number of classes : " + str(len(set(Y))) + "\n\n")
    text.insert(END, "Class labels found in dataset : " + str(disease) + "\n")       

def trainTumorDetectionModel():
    global accuracy
    global classifier
    global X
    global Y

    if len(X) == 0:
        if os.path.exists('Model/myimg_data.txt.npy'):
            X = np.load('Model/myimg_data.txt.npy')
            Y = np.load('Model/myimg_label.txt.npy')
        else:
            messagebox.showinfo("Error", "Please preprocess dataset first!")
            return

    YY = to_categorical(Y)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    x_train = X[indices]
    y_train = YY[indices]

    if os.path.exists('Model/model.json'):
        with open('Model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)

        classifier.load_weights("Model/model_weights.h5")
        if hasattr(classifier, '_make_predict_function'):
            classifier._make_predict_function()
        elif hasattr(classifier, 'make_predict_function'):
            classifier.make_predict_function()           
    else:
        X_trains, X_tests, y_trains, y_tests = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
        classifier = Sequential() 
        classifier.add(Conv2D(32, (3, 3), input_shape=(128, 128, 1), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Conv2D(32, (3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(128, activation='relu'))
        classifier.add(Dense(2, activation='softmax'))
        print(classifier.summary())
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        hist = classifier.fit(x_train, y_train, batch_size=16, epochs=10, validation_split=0.2, shuffle=True, verbose=2)
        classifier.save_weights('Model/model_weights.h5')            
        model_json = classifier.to_json()
        with open("Model/model.json", "w") as json_file:
            json_file.write(model_json)
        with open('Model/history.pckl', 'wb') as f:
            pickle.dump(hist.history, f)

    history_path = 'Model/history.pckl' if os.path.exists('Model/history.pckl') else 'model/history.pckl'
    with open(history_path, 'rb') as f:
        data = pickle.load(f)
    acc = data['accuracy']
    accuracy = acc[4] * 100 if len(acc) > 4 else acc[-1] * 100
    text.insert(END, '\n\nCNN Brain Tumor Model Generated. See console to view layers of CNN\n\n')
    text.insert(END, "CNN Brain Tumor Prediction Accuracy on Test Images : " + str(accuracy) + "\n")

def ensureClassifierLoaded():
    global classifier
    if classifier is None:
        if os.path.exists('Model/model.json') and os.path.exists('Model/model_weights.h5'):
            with open('Model/model.json', "r") as json_file:
                classifier = model_from_json(json_file.read())
            classifier.load_weights("Model/model_weights.h5")
            if hasattr(classifier, '_make_predict_function'):
                classifier._make_predict_function()
            elif hasattr(classifier, 'make_predict_function'):
                classifier.make_predict_function()
            return True
        else:
            return False
    return True

def tumorClassification():
    global classifier
    if not ensureClassifierLoaded():
        messagebox.showinfo("Error", "Please train the CNN Brain Tumor Detection Model first!")
        return

    test_file = filedialog.askopenfilename(initialdir="testImages", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.JPG;*.PNG")])
    if not test_file:
        return

    img = cv2.imread(test_file, 0)
    if img is None:
        messagebox.showinfo("Error", "Could not read the selected image.")
        return

    img_res = cv2.resize(img, (128, 128))
    im2arr = np.array(img_res)
    im2arr = im2arr.reshape(1, 128, 128, 1)
    XX = np.asarray(im2arr)
        
    predicts = classifier.predict(XX)
    print("Prediction probabilities:", predicts)
    cls = np.argmax(predicts)
    print("Class:", cls, "-", disease[cls])

    text.insert(END, "\nTesting image: " + os.path.basename(test_file) + "\n")
    text.insert(END, "Classification Result : " + disease[cls] + "\n")

    if cls == 0:
        disp_img = cv2.imread(test_file)
        disp_img = cv2.resize(disp_img, (800, 500))
        cv2.putText(disp_img, 'Classification Result : ' + disease[cls], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Classification Result : ' + disease[cls], disp_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if cls == 1:
        segmented_image, edge_image = tumorSegmentation(test_file)
        disp_img = cv2.imread(test_file)
        disp_img = cv2.resize(disp_img, (800, 500))
        cv2.putText(disp_img, 'Classification Result : ' + disease[cls], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Classification Result : ' + disease[cls], disp_img)
        cv2.imshow("Tumor Segmented Image", segmented_image)
        cv2.imshow("Edge Detected Image", edge_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def graph():
    history_path = 'Model/history.pckl' if os.path.exists('Model/history.pckl') else 'model/history.pckl'
    if not os.path.exists(history_path):
        messagebox.showinfo("Error", "Model history not found. Please train model first.")
        return
    with open(history_path, 'rb') as f:
        data = pickle.load(f)

    accuracy_vals = data['accuracy']
    loss_vals = data['loss']

    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.xlabel('Training Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss_vals, 'ro-', color='red', label='Loss')
    plt.plot(accuracy_vals, 'ro-', color='green', label='Accuracy')
    plt.legend(loc='upper left')
    plt.title('Brain Tumor CNN Model Training Accuracy & Loss Graph')
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='Identifying Brain Tumor using X-Ray Images')
title.config(bg='darkviolet', fg='gold')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=150)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=120)
text.config(font=font1)

uploadButton = Button(main, text="Upload Tumor X-Ray Images Dataset", command=uploadDataset)
uploadButton.place(x=50, y=550)
uploadButton.config(font=font1)  

preprocessButton = Button(main, text="Dataset Preprocessing & Features Extraction", command=datasetPreprocessing)
preprocessButton.place(x=430, y=550)
preprocessButton.config(font=font1) 

cnnButton = Button(main, text="Trained CNN Brain Tumor Detection Model", command=trainTumorDetectionModel)
cnnButton.place(x=810, y=550)
cnnButton.config(font=font1) 

classifyButton = Button(main, text="Brain Tumor Segmentation & Classification", command=tumorClassification)
classifyButton.place(x=50, y=600)
classifyButton.config(font=font1)

graphButton = Button(main, text="Training Accuracy Graph", command=graph)
graphButton.place(x=430, y=600)
graphButton.config(font=font1)

main.config(bg='turquoise')
main.mainloop()
