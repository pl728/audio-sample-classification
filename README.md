In [1]:

    import librosa
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout

In [2]:

    # parse the kicks, crashes 
    # 0 = kick, 1 = crash

    feature = []
    label = []

    for i in range(1, 202):
        d, sr = librosa.load("sample_kick/VEH1 Hard Kick - " + str(i).zfill(3) + ".wav", sr=44100, res_type='kaiser_fast')
        mels = np.mean(librosa.feature.melspectrogram(y=d, sr=sr).T, axis=0)
        feature.append(mels)
        label.append(0)

    for i in range(1, 51):
        d, sr = librosa.load("sample_crash/VEH1 Crash - " + str(i).zfill(2) + ".wav", sr=44100, res_type='kaiser_fast')
        mels = np.mean(librosa.feature.melspectrogram(y=d, sr=sr).T, axis=0)
        feature.append(mels)
        label.append(1)
        
    print(feature[0].shape)
        


    data = {
        "X": np.array(feature),
        "t": np.array(label)
    }

    data["t"] = tf.keras.utils.to_categorical(data["t"])
    print(data["X"])

    (128,)
    [[4.9364899e+02 1.3382242e+03 8.3680701e+02 ... 8.0354203e-06
      8.5975907e-06 6.0973957e-06]
     [2.1349692e+02 9.8276776e+02 5.0937537e+02 ... 1.7508426e-03
      2.7589572e-03 8.0489012e-04]
     [3.4847284e+02 9.0860150e+02 3.3291241e+02 ... 4.1663774e-05
      7.3597308e-05 2.4802917e-05]
     ...
     [2.2502916e-05 8.3084352e-04 2.2441533e-03 ... 4.6322175e-08
      4.7188419e-08 5.0364477e-08]
     [2.0987345e-02 6.6036671e-02 1.5265882e-01 ... 2.5161073e-06
      2.5729428e-06 2.3973475e-06]
     [7.0509864e-03 2.2733379e-02 3.5233915e-02 ... 7.3609786e-04
      7.8171847e-04 1.1218864e-03]]

In [3]:

    X_train, X_test, Y_train, Y_test = train_test_split(data["X"], data["t"], random_state=1)
    print(X_train.shape)
    X_train = X_train.reshape(188, 16, 8, 1)
    X_test = X_test.reshape(63, 16, 8, 1)
    print(X_train.shape)

    (188, 128)
    (188, 16, 8, 1)

In [4]:

    input_dim = (16, 8, 1)

In [5]:

    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding = "same", activation = "tanh", input_shape = input_dim))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding = "same", activation = "tanh"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(1024, activation = "tanh"))
    model.add(Dense(2, activation = "softmax"))

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    model.fit(X_train, Y_train, epochs = 3, batch_size = 1, validation_data = (X_test, Y_test))
    model.summary()

    Train on 188 samples, validate on 63 samples
    Epoch 1/3
    188/188 [==============================] - 4s 20ms/sample - loss: 0.3363 - accuracy: 0.9787 - val_loss: 3.6908e-04 - val_accuracy: 1.0000
    Epoch 2/3
    188/188 [==============================] - 3s 17ms/sample - loss: 6.0605e-05 - accuracy: 1.0000 - val_loss: 3.2526e-04 - val_accuracy: 1.0000
    Epoch 3/3
    188/188 [==============================] - 3s 18ms/sample - loss: 6.7195e-05 - accuracy: 1.0000 - val_loss: 2.0428e-04 - val_accuracy: 1.0000
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 16, 8, 64)         640       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 8, 4, 64)          0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 8, 4, 128)         73856     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 4, 2, 128)         0         
    _________________________________________________________________
    dropout (Dropout)            (None, 4, 2, 128)         0         
    _________________________________________________________________
    flatten (Flatten)            (None, 1024)              0         
    _________________________________________________________________
    dense (Dense)                (None, 1024)              1049600   
    _________________________________________________________________
    dense_1 (Dense)              (None, 2)                 2050      
    =================================================================
    Total params: 1,126,146
    Trainable params: 1,126,146
    Non-trainable params: 0
    _________________________________________________________________

In [6]:

    # predictions = model.predict(X_test)
    score = model.evaluate(X_test, Y_test)
    print(score)

    63/63 [==============================] - 0s 5ms/sample - loss: 2.0428e-04 - accuracy: 1.0000
    [0.0002042830337069561, 1.0]

In [7]:

    model.save("kick-crash-classifier")
