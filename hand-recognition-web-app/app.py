
import streamlit as st
from streamlit_option_menu import option_menu
import csv
import mediapipe as mp
import pandas as pd
import cv2
import numpy as np
import copy
import itertools
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import random


def main():
    ##mediapipe solutions
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html= True)
    with st.sidebar:
        select = option_menu(
            menu_title=None,
            options=['Home','Collect Data Points', 'Train','Inference'],
            default_index= 0
        )
        
    #reading label
    with open('label.csv', encoding='utf-8-sig') as f:
        keypoint_labels = csv.reader(f)
        keypoint_labels = [row[0] for row in keypoint_labels]    
        
    if select == 'Home':
        st.title('Hand Gesture Recognition web-app')
        
        
    if select == 'Collect Data Points':
        st.markdown("### Capture Data Points")
        ##label input
        st.sidebar.markdown('---', unsafe_allow_html=True)
        dont_write = ['']
        write_labels = st.sidebar.checkbox('write labels')
        if write_labels:
            text = st.sidebar.text_input('Input labels')
            text = text.split(",")
            print(text)
            if text != dont_write:
                loglabels(text)
            # if len(text) >= 1:
        
        
        ##camera capture
        run_camera = st.sidebar.checkbox('Start capture')
        # cap = cv2.VideoCapture(0)
        # FRAME_WINDOW = st.image([])
        
        shift_intensity = st.sidebar.slider('shift intensity',1,10,step=int(1))
        num_of_shift = int(st.sidebar.number_input('number of augmented data points'))
        if run_camera:
            capture = st.button('CAPTURE')
            #reading label
            with open('label.csv', encoding='utf-8-sig') as f:
                keypoint_labels = csv.reader(f)
                keypoint_labels = [row[0] for row in keypoint_labels]
            ##creating a select box
            number = keypoint_labels.index(st.selectbox('Capturing for..',keypoint_labels))
            
            ##augmenting parameters

            cap = cv2.VideoCapture(0)
            FRAME_WINDOW = st.image([])
                
            with mp_hands.Hands(model_complexity = 0, min_detection_confidence = 0.7, min_tracking_confidence = 0.5) as hands:
                while cap.isOpened():
                    ret, frame = cap.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.flip(frame,1)
                    results = hands.process(frame)
                    ##drawing hand anotations
                    frame.flags.writeable =True
                    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    if results.multi_hand_landmarks:
                        for hand_landmarks,handedness in zip(results.multi_hand_landmarks,results.multi_handedness):
                            mp_drawing.draw_landmarks(frame,
                            hand_landmarks,mp_hands.HAND_CONNECTIONS)
                            #drawing bounds
                            rect_coordinates = calculating_bound(frame,hand_landmarks)
                            
                            cv2.rectangle(frame,(rect_coordinates[0],rect_coordinates[1]),(rect_coordinates[2],rect_coordinates[3]),(255,123,50),2)
                            ##adding handedness 
                            info_text(frame,handedness,rect_coordinates)
                            ##capturing landmarks
                            handlandmarks = find_position(frame,hand_landmarks)
                            
                            if capture:
                                pre_processing_lms(number,handlandmarks,num_of_shift,shift_intensity)
                                # logginglandmarks(number,preprocessed_landmarks)
                                capture = False
                    FRAME_WINDOW.image(frame,width=450)
                    
    if select ==  'Train':
        st.markdown("# Training",unsafe_allow_html=True)
        num_classes = len(keypoint_labels)
        st.write('Number of classes:',num_classes)
        ## reading data points
        dataframe = pd.read_csv("landmarks.csv", header=None)
        x = dataframe.iloc[:,1:43].values
        y = dataframe.iloc[:,0].values
        with st.expander('View Data Frames'):
            st.text('Dataframe')
            st.dataframe(dataframe)

def dataPreprocessing(x,y):
    ##Normalizing handlandmarks between 0 and 1
    sc = MinMaxScaler(feature_range=(0,1))
    x_scaled =sc.fit_transform(x)
    ##Splitting handlandmarks to training and test sets
    x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,test_size=0.2,random_state=0)
    return x_train,x_test,y_train,y_test

def build_nn():
    pass
    
            
    
        
        
        
##writing labels to csv
def loglabels(text):
    csv_path = 'label.csv'
    with open(csv_path, 'w', newline="") as f:
        writer = csv.writer(f, delimiter="\n")
        writer.writerow(text)
        
##writing landmark labels
def logginglandmarks(number, landmarks):
    csv_path = 'landmarks.csv'
    with open(csv_path,'a',newline="") as f:
        writer = csv.writer(f)
        writer.writerow([number, *landmarks])
    return

#Calculating the rectangle bounds for hands   
def calculating_bound(frame,landmarks):
    frame_height,frame_width = frame.shape[0],frame.shape[1]
    landmark_array = np.empty((0,2), int)
    for _,lms in enumerate(landmarks.landmark):
        x_val = min(int(lms.x * frame_width), frame_width)
        y_val = min(int(lms.y * frame_height),frame_height)
        landmark_point = [np.array((x_val,y_val))]
        landmark_array =np.append(landmark_array,landmark_point, axis=0)
        x,y,width,height = cv2.boundingRect(landmark_array)
    return [x,y,width+x,height+y]

#collecting landmarks into an array
def find_position(frame,landmarks):
    landmark_points = []
    for _,lms in enumerate(landmarks.landmark):
        #getting the dimensions of the capture
        h,w,c = frame.shape #returns height, width, channel number
        #changing the values of landmarks into pixels
        cap_x,cap_y = min(int(lms.x*w),w-1),min(int(lms.y*h),h-1)
        landmark_points.append([cap_x,cap_y])
    return np.asarray(landmark_points)

##preprocessing the landmarks
def pre_processing_lms(number,landmarklist,num_shift,shift_intensity):
        temp_lms = copy.deepcopy(landmarklist)
        # print(type(temp_lms))
        #getting the relative coordinates
        base_x,base_y = 0,0
        for index, landmark_point in enumerate(temp_lms):
            if index == 0:
                base_x,base_y = landmark_point[0], landmark_point[1]
                
            temp_lms[index][0] = temp_lms[index][0] - base_x
            temp_lms[index][1] = temp_lms[index][1] - base_y
        # print(type(temp_lms))
        augment(number,temp_lms,num_shift,shift_intensity)
        

##function to augment landmark points
def augment(number,landmark,num_shift,shift_intensity):
    #we are calling funtion inside the while loop
    
    
    def _shift_diagnol_up(number,landmark):
        for x in range(num_shift):
            new_point = np.reshape(random.randint(-50,50), (1, 1)) + landmark
            writeable = list(itertools.chain.from_iterable(new_point))
            logginglandmarks(number,writeable)
            landmark = new_point
            
    def _shift_diagnol_down(number,landmark):
        for x in range(num_shift):
            new_point = np.reshape(-random.randint(-50,50), (1, 1)) + landmark
            writeable = list(itertools.chain.from_iterable(new_point))
            logginglandmarks(number,writeable)
            landmark = new_point
            
    def _shift_right(number,landmark):
        for x in range(num_shift):
            new_point = [[random.randint(-50,50),0]]+ landmark
            writeable = list(itertools.chain.from_iterable(new_point))
            logginglandmarks(number,writeable)
            landmark = new_point


    def _shift_left(number,landmark):
        for x in range(num_shift):
            new_point = [[random.randint(-50,50), 0]] + landmark
            ##writing into csv target file
            writeable = list(itertools.chain.from_iterable(new_point))
            logginglandmarks(number,writeable)
            landmark = new_point
            
    def _shift_up(number,landmark):
        # if number >-1:
        for x in range(num_shift):
            new_point = [[0,-random.randint(-50,50)]] + landmark
            ##writing into csv target file
            writeable = list(itertools.chain.from_iterable(new_point))
            logginglandmarks(number,writeable)
            landmark = new_point
    def _shift_down(number,landmark):
        # if number >-1:
        for x in range(num_shift):
            new_point = [[0,random.randint(-50,50)]] + landmark
            ##writing into csv target file
            writeable = list(itertools.chain.from_iterable(new_point))
            logginglandmarks(number,writeable)
            landmark = new_point
                
    
    if number >= 0:       
        _shift_down(number,landmark)
        _shift_up(number,landmark)
        _shift_left(number,landmark)
        _shift_right(number,landmark)
        _shift_diagnol_down(number,landmark)
        _shift_diagnol_up(number,landmark)
    
    # _shift_diagnol_down(landmark)
    # _shift_diagnol_up(landmark)


##getting the info text
def info_text(frame, handedness,rect_coordinates):
    info_text = handedness.classification[0].label
    # if hand_sign_text != "":
    #     info_text = info_text + ":" + hand_sign_text
    cv2.putText(frame,info_text,(rect_coordinates[0],rect_coordinates[1]-22),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
    return frame


    

if __name__ == '__main__':
    main()