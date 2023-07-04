import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import requests
from zipfile import ZipFile
import sys

if __name__ == '__main__':

    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    if not os.path.exists('../SavedModels'):
        os.mkdir('../SavedModels')

    if not os.path.exists('../SavedHistory'):
        os.mkdir('../SavedHistory')

    # Download data if it is unavailable.
    if 'cats-and-dogs-images.zip' not in os.listdir('../Data'):
        sys.stderr.write("[INFO] Image dataset is loading.\n")
        url = "https://www.dropbox.com/s/jgv5zpw41ydtfww/cats-and-dogs-images.zip?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/cats-and-dogs-images.zip', 'wb').write(r.content)
        sys.stderr.write("[INFO] Loaded.\n")

        sys.stderr.write("\n[INFO] Extracting files.\n")
        with ZipFile('../Data/cats-and-dogs-images.zip', 'r') as myzip:
            myzip.extractall(path="../Data")
            sys.stderr.write("[INFO] Completed.\n")

    # write your code here
    from keras.applications.vgg16 import preprocess_input
    from keras.preprocessing.image import ImageDataGenerator
    from keras.applications.vgg16 import VGG16
    from keras import Sequential
    from keras.layers import Dense
    import tensorflow as tf
    import pickle
    import numpy as np

    # Hyper parameters

    size_train = 500  # Number of files in the directory
    size_valid = 200  # Number of files in the directory
    image_height = 150
    image_width = 150
    epochs = 5
    learning_rate = 1e-3


    # File paths
    model_path = "../SavedModels/"
    history_path = "../SavedHistory/"
    predictions_path = "../SavedHistory/"


    def generate_sets(_image_height, _image_width, _batch_size):
        dn = ImageDataGenerator(preprocessing_function=preprocess_input)

        dn_train = dn.flow_from_directory(
            target_size=(_image_height, _image_width),
            batch_size=_batch_size,
            class_mode="categorical",
            directory="../Data/train",
            shuffle=False,
        )
        dn_valid = dn.flow_from_directory(
            target_size=(_image_height, _image_width),
            batch_size=_batch_size,
            class_mode="categorical",
            directory="../Data/valid",
            shuffle=False,
        )
        dn_test = dn.flow_from_directory(
            target_size=(_image_height, _image_width),
            batch_size=_batch_size,
            class_mode="categorical",
            directory="../Data/",
            classes=["test"],
            shuffle=False,
        )

        print(
            dn_test.image_shape[0],
            dn_test.image_shape[1],
            dn_test.batch_size,
            dn_test.shuffle,
        )

        return dn_train, dn_valid, dn_test


    def create_save_model(_dn_train, _dn_valid, _size_train, _size_valid, _file_name):

        # Create a model
        model = Sequential(
            [
                VGG16(
                    include_top=False,
                    pooling='avg',
                    weights='imagenet'),
                Dense(2, activation='softmax'),
            ]
        )
        model.layers[0].trainable = False

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

        # Fit the model
        history = model.fit(
            x=_dn_train,
            epochs=epochs,
            validation_data=_dn_valid,
            steps_per_epoch=_size_train // _dn_train.batch_size,
            validation_steps=_size_valid // _dn_valid.batch_size,
            verbose=1,
        )

        # Save the model
        model.save(f"../SavedModels/{_file_name}.h5")

        # Save model history as pickle
        with open(f"../SavedHistory/{_file_name}", 'wb') as handle:
            pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return model, history


    def create_save_predictions(_model, _dn_test, _file_name):
        # model = tf.keras.models.load_model(model_path)
        predictions = _model.predict(
            x=_dn_test,
            batch_size=None,
            verbose='auto',
            steps=None,
            callbacks=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False
        )
        predictions_actual_classes = np.argmax(
            a=predictions,
            axis=1,
        )
        print(predictions_actual_classes)

        with open(f"../SavedHistory/{_file_name}", 'wb') as handle:
            pickle.dump(predictions_actual_classes, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return predictions_actual_classes


    def stage4(heights, widths, batch_sizes, file_names):
        prev_accuracy = float("-inf")
        accuracies = []

        for height, width, bsize, file_name in zip(heights, widths, batch_sizes, file_names):
            dn_train, dn_valid, dn_test = generate_sets(height, width, bsize)
            model, history = create_save_model(
                dn_train,
                dn_valid,
                size_train,
                size_valid,
                file_name
            )

            predictions_actual_classes = create_save_predictions(
                model,
                dn_test,
                file_name
            )

            accuracy = history.history['accuracy'][-1]
            print("accuracy:", accuracy)

            if accuracy > prev_accuracy:
                print("accuracy update:", accuracy)
                # save models and history
                model.save(f"../SavedModels/stage_four_model.h5")
                with open(f"../SavedHistory/stage_four_history", 'wb') as handle:
                    pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(f"../SavedHistory/stage_four_predictions", 'wb') as handle:
                    pickle.dump(predictions_actual_classes, handle, protocol=pickle.HIGHEST_PROTOCOL)
            prev_accuracy = accuracy

            accuracies.append([file_name, accuracy])
        return accuracies


    # result = stage4(
    #     heights=(150, 200, 224),
    #     widths=(150, 200, 224),
    #     batch_sizes=(16, 32, 64),
    #     file_names=("file1", "file2", "file3"),
    # )
    # Test result
    # [[1, 0.9235537052154541, 150, 150, 64], [2, 0.9316239356994629, 200, 200, 64], [3, 0.85550457239151, 224, 224, 64]]

    def stage5():
        # best hyper params found in the previous stage
        height = 200
        width = 200
        batch_size = 64
        global learning_rate
        learning_rate = 1e-5
        file_name_model = f"../SavedModels/stage_four_model.h5"
        file_name_history = "stage_five_history"
        prev_accuracy = float("-inf")
        accuracies = []

        dn_train, dn_valid, dn_test = generate_sets(height, width, batch_size)
        model = tf.keras.models.load_model(f"../SavedModels/stage_four_model.h5")

        # Find tune your model by making last few layers trainable
        last_n = 3
        for n in range(1, last_n + 1):
            for layer in model.layers[0].layers[-n:]:
                print(n)
                layer.trainable = True

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy'],
            )

            history = model.fit(
                x=dn_train,
                epochs=epochs,
                validation_data=dn_valid,
                steps_per_epoch=size_train // dn_train.batch_size,
                validation_steps=size_valid // dn_valid.batch_size,
                verbose=1,
            )

            predictions = model.predict(
                x=dn_test,
                batch_size=None,
                verbose='auto',
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False
            )

            predictions_actual_classes = np.argmax(
                a=predictions,
                axis=1,
            )

            # first 25 items are cat (class 0), next 25 items are dog (class 1)
            correct_predictions = sum([1 for v in predictions_actual_classes[:25] if not v]) + sum(
                [1 for v in predictions_actual_classes[25:] if v])
            rate = correct_predictions / len(predictions_actual_classes)
            print("rate :", rate)
            accuracy = history.history['accuracy'][-1]
            print("accuracy :", accuracy)

            if accuracy > prev_accuracy:
                print("accuracy update:", accuracy)
                model.save(f"../SavedModels/{file_name_model}")
                with open(f"../SavedHistory/{file_name_history}", 'wb') as handle:
                    pickle.dump(predictions_actual_classes, handle, protocol=pickle.HIGHEST_PROTOCOL)
            prev_accuracy = accuracy
            accuracies.append([accuracy])
        return accuracies

    print(stage5())
