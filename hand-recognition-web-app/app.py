###--------------------------------Imports---------------------------------------------------------
import copy
import csv
import itertools
import random
import time

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from keypoint_classifier import KeyPointClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from streamlit_option_menu import option_menu


def main():
    ##mediapipe solutions
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    saved_model_path = "models/savedkeypoints.h5"
    tflite_model_path = "models/savedkeypointclassifies.tflite"

    FRAME_WINDOW = st.image([])

    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    with st.sidebar:
        select = option_menu(
            menu_title=None,
            options=["Home", "Collect Data Points", "Train", "Inference"],
            default_index=0,
        )

    # reading label
    with open("label.csv", encoding="utf-8-sig") as f:
        keypoint_labels = csv.reader(f)
        keypoint_labels = [row[0] for row in keypoint_labels]

    if select == "Home":
        st.title("Hand Gesture Recognition web-app")

    with open("label.csv", encoding="utf-8-sig") as f:
        keypoint_labels = csv.reader(f)
        keypoint_labels = [row[0] for row in keypoint_labels]

    # reading dataframe from csv file
    dataframe = pd.read_csv("keypoints.csv", header=None)
    x = dataframe.iloc[:, 1:43].values
    x = pd.DataFrame(x)
    y = dataframe[0]

    # splitting dataset
    x_train, x_test, y_train, y_test = dataPreprocessing(x, y)

    # performing pearsons correlation on x_train
    corr_features = correlation(x_train, 0.85)
    print(corr_features)

    if select == "Collect Data Points":
        st.markdown("### Capture Data Points")
        ##label input
        st.sidebar.markdown("---", unsafe_allow_html=True)
        dont_write = [""]
        write_labels = st.sidebar.checkbox("write labels")
        if write_labels:
            text = st.sidebar.text_input("Input labels")
            text = text.split(",")
            print(text)
            if text != dont_write:
                loglabels(text)
            # if len(text) >= 1:

        ##camera capture
        run_camera = st.sidebar.checkbox("Start capture")
        # cap = cv2.VideoCapture(0)
        # FRAME_WINDOW = st.image([])

        # shift_intensity = st.sidebar.slider('shift intensity',1,10,step=int(1))
        num_of_shift = int(st.sidebar.number_input("number of augmented data points"))
        if run_camera:

            capture = st.button("CAPTURE")
            ##creating a select box
            number = keypoint_labels.index(
                st.selectbox("Capturing for..", keypoint_labels)
            )

            ##augmenting parameters

            cap = cv2.VideoCapture(0)
            with mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            ) as hands:
                while cap.isOpened():
                    ret, frame = cap.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.flip(frame, 1)
                    results = hands.process(frame)
                    ##drawing hand anotations
                    frame.flags.writeable = True
                    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    if results.multi_hand_landmarks:
                        for hand_landmarks, handedness in zip(
                            results.multi_hand_landmarks, results.multi_handedness
                        ):
                            mp_drawing.draw_landmarks(
                                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                            )
                            # drawing bounds
                            rect_coordinates = calculating_bound(frame, hand_landmarks)

                            cv2.rectangle(
                                frame,
                                (rect_coordinates[0], rect_coordinates[1]),
                                (rect_coordinates[2], rect_coordinates[3]),
                                (255, 123, 50),
                                2,
                            )
                            ##adding handedness
                            info_text(frame, handedness, rect_coordinates)
                            ##capturing landmarks
                            handlandmarks = find_position(frame, hand_landmarks)

                            if capture:
                                pre_processing_lms(number, handlandmarks, num_of_shift)
                                # logginglandmarks(number,preprocessed_landmarks)
                                capture = False
                    FRAME_WINDOW.image(frame, width=450)

    if select == "Train":
        classifier = Sequential()
        st.markdown("# Training", unsafe_allow_html=True)
        num_classes = len(keypoint_labels)
        st.write("Number of classes:", num_classes)
        ## reading data points
        with st.expander("View Data Frames"):
            ##sumarize the df
            st.text("Dataframe")
            st.dataframe(dataframe)
            st.dataframe(x_train)
        ##preprocess data
        # x_train, x_test, y_train, y_test = dataPreprocessing(x, y)
        x_train = x_train.drop(corr_features, axis=1)
        x_test = x_test.drop(corr_features, axis=1)
        ##Normalizing handlandmarks between 0 and 1
        sc = MinMaxScaler(feature_range=(0, 1))
        x_train = sc.fit_transform(x_train)
        x_test = sc.fit_transform(x_test)

        ##build and compile
        build_nn(num_classes, classifier)
        cp_callback = ModelCheckpoint(
            saved_model_path, verbose=1, save_weights_only=False
        )
        es_callback = EarlyStopping(patience=20, verbose=1)
        classifier.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        ##train preprocessed data
        if st.button("TRAIN"):
            classifier.fit(
                x_train,
                y_train,
                epochs=100,
                batch_size=32,
                callbacks=[cp_callback, es_callback],
                validation_data=(x_test, y_test),
            )
            Accuracy = classifier.evaluate(x_test, y_test)
            st.write("Training Successful with Accuracy:", Accuracy[1])

        # ## saving the model
        model = load_model(saved_model_path)
        # ##making predictions
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=-1)
        # Work on showing confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plot_cm = st.sidebar.checkbox("Plot Confusion Matrix")
        if plot_cm:
            plot_confusion_matrix(cm, classes=keypoint_labels)
        #     # st.pyplot()

        # model.save(saved_model_path)
        save_tflite = st.button("Save tflite")
        if save_tflite:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            open(tflite_model_path, "wb").write(tflite_model)
            save_tflite = False

    if select == "Inference":
        keypoint_classifier = KeyPointClassifier()

        Infere = st.sidebar.checkbox("Start Inference")
        if Infere:
            cap = cv2.VideoCapture(0)
            with mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            ) as hands:
                while cap.isOpened():
                    ret, frame = cap.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.flip(frame, 1)
                    results = hands.process(frame)
                    ##drawing hand anotations
                    frame.flags.writeable = True
                    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    if results.multi_hand_landmarks:
                        for hand_landmarks, handedness in zip(
                            results.multi_hand_landmarks, results.multi_handedness
                        ):
                            mp_drawing.draw_landmarks(
                                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                            )
                            # drawing bounds
                            rect_coordinates = calculating_bound(frame, hand_landmarks)

                            cv2.rectangle(
                                frame,
                                (rect_coordinates[0], rect_coordinates[1]),
                                (rect_coordinates[2], rect_coordinates[3]),
                                (255, 123, 50),
                                2,
                            )
                            ##capturing landmarks
                            handlandmarks = find_position(frame, hand_landmarks)
                            processed_landmark_list = inference_lms(handlandmarks)
                            processed_landmark_list = pd.DataFrame(
                                [processed_landmark_list]
                            )
                            processed_landmark_list = processed_landmark_list.drop(
                                corr_features, axis=1
                            )
                            print(type(processed_landmark_list))
                            processed_landmark_list = list(
                                itertools.chain.from_iterable(
                                    processed_landmark_list.values.tolist()
                                )
                            )
                            print(processed_landmark_list)
                            # collecting hand sign index
                            hand_sign_id = keypoint_classifier(processed_landmark_list)
                            ##adding info text
                            inference_text(
                                frame,
                                handedness,
                                rect_coordinates,
                                keypoint_labels[hand_sign_id],
                            )
                    FRAME_WINDOW.image(frame, width=450)


##---------------------------------------DATAPROCESSING AND BUILDING NEURAL NETWORK-----------------------------
def dataPreprocessing(x, y):
    ##Normalizing handlandmarks between 0 and 1
    # sc = MinMaxScaler(feature_range=(0, 1))
    # x_scaled = sc.fit_transform(x)
    ##Splitting handlandmarks to training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0
    )
    return x_train, x_test, y_train, y_test


def build_nn(num_classes, classifier):
    classifier.add(Dense(42, activation="relu", input_shape=(7,)))
    classifier.add(Dense(32, activation="relu"))
    classifier.add(Dense(num_classes, activation="softmax"))


# pearson correlation
def correlation(dataset, threshold):
    col_cor = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_cor.add(colname)
    return col_cor


###------------------------------------------CONFUSION MATRIX PLOT--------------------------------------------------


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig, ax = plt.subplots()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    st.pyplot(fig)


##-------------------------------------------GETTING AND WRITING LMS ---------------------------------------------------
##writing labels to csv
def loglabels(text):
    csv_path = "label.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\n")
        writer.writerow(text)


##writing landmark labels
def logginglandmarks(number, landmarks):
    csv_path = "landmarks.csv"
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([number, *landmarks])
    return


# Calculating the rectangle bounds for hands
def calculating_bound(frame, landmarks):
    frame_height, frame_width = frame.shape[0], frame.shape[1]
    landmark_array = np.empty((0, 2), int)
    for _, lms in enumerate(landmarks.landmark):
        x_val = min(int(lms.x * frame_width), frame_width)
        y_val = min(int(lms.y * frame_height), frame_height)
        landmark_point = [np.array((x_val, y_val))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
        x, y, width, height = cv2.boundingRect(landmark_array)
    return [x, y, width + x, height + y]


# collecting landmarks into an array
def find_position(frame, landmarks):
    landmark_points = []
    for _, lms in enumerate(landmarks.landmark):
        # getting the dimensions of the capture
        h, w, c = frame.shape  # returns height, width, channel number
        # changing the values of landmarks into pixels
        cap_x, cap_y = min(int(lms.x * w), w - 1), min(int(lms.y * h), h - 1)
        landmark_points.append([cap_x, cap_y])
    return np.asarray(landmark_points)


##preprocessing the landmarks
def pre_processing_lms(number, landmarklist, num_shift):
    temp_lms = copy.deepcopy(landmarklist)
    # print(type(temp_lms))
    # getting the relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_lms):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_lms[index][0] = temp_lms[index][0] - base_x
        temp_lms[index][1] = temp_lms[index][1] - base_y

    augment(number, temp_lms, num_shift)


def inference_lms(landmarklist):
    temp_lms = copy.deepcopy(landmarklist)
    # getting the relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_lms):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_lms[index][0] = temp_lms[index][0] - base_x
        temp_lms[index][1] = temp_lms[index][1] - base_y

    # converting the array to a one-demension. itertools.chain returns elements of lists until loop is exhausted
    temp_lms = list(itertools.chain.from_iterable(temp_lms))
    return temp_lms


##function to augment landmark points
def augment(number, landmark, num_shift):
    def _shift_diagnol_up(number, landmark):
        for x in range(num_shift):
            new_point = np.reshape(random.randint(-100, 100), (1, 1)) + landmark
            writeable = list(itertools.chain.from_iterable(new_point))
            logginglandmarks(number, writeable)
            landmark = new_point

    def _shift_diagnol_down(number, landmark):
        for x in range(num_shift):
            new_point = np.reshape(-random.randint(-100, 100), (1, 1)) + landmark
            writeable = list(itertools.chain.from_iterable(new_point))
            logginglandmarks(number, writeable)
            landmark = new_point

    def _shift_right(number, landmark):
        for x in range(num_shift):
            new_point = [[random.randint(-50, 50), 0]] + landmark
            writeable = list(itertools.chain.from_iterable(new_point))
            logginglandmarks(number, writeable)
            landmark = new_point

    def _shift_left(number, landmark):
        for x in range(num_shift):
            new_point = [[random.randint(-50, 50), 0]] + landmark
            ##writing into csv target file
            writeable = list(itertools.chain.from_iterable(new_point))
            logginglandmarks(number, writeable)
            landmark = new_point

    def _shift_up(number, landmark):
        # if number >-1:
        for x in range(num_shift):
            new_point = [[0, -random.randint(-50, 50)]] + landmark
            ##writing into csv target file
            writeable = list(itertools.chain.from_iterable(new_point))
            logginglandmarks(number, writeable)
            landmark = new_point

    def _shift_down(number, landmark):
        for x in range(num_shift):
            new_point = [[0, random.randint(-50, 50)]] + landmark
            ##writing into csv target file
            writeable = list(itertools.chain.from_iterable(new_point))
            logginglandmarks(number, writeable)
            landmark = new_point

    if number >= 0:
        _shift_down(number, landmark)
        _shift_up(number, landmark)
        _shift_left(number, landmark)
        _shift_right(number, landmark)
        _shift_diagnol_down(number, landmark)
        _shift_diagnol_up(number, landmark)

    # _shift_diagnol_down(landmark)
    # _shift_diagnol_up(landmark)


##getting the info text
def info_text(frame, handedness, rect_coordinates):
    info_text = handedness.classification[0].label
    cv2.putText(
        frame,
        info_text,
        (rect_coordinates[0], rect_coordinates[1] - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
    )
    return frame


def inference_text(frame, handedness, rect_coordinates, hand_sign_text):
    info_text = handedness.classification[0].label
    info_text = info_text + ":" + hand_sign_text
    cv2.putText(
        frame,
        info_text,
        (rect_coordinates[0], rect_coordinates[1] - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
    )
    return frame


if __name__ == "__main__":
    main()
