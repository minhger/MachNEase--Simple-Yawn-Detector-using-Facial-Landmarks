# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 23:09:35 2022

@author: nguye
"""

import cv2
import numpy as np
import dlib

detector = dlib.get_frontal_face_detector() #For detecting faces
landmark_path="shape_predictor_68_face_landmarks.dat" #Path of the file - if stored in the same directory. Else, give the relative path
predictor = dlib.shape_predictor(landmark_path) #For identifying landmarks

#Obtaining Facial Landmark coordinates
def get_facial_landmarks(image):
    face = detector(image, 1)
    #Detecting faces in image
    if len(face) > 1:
        return "Multiple faces detected in the frame!!"
    if len(face) == 0:
        return "No face detected in the frame!!"
    #Return the coordinates
    #Predictor identifies all the 68 landmarks for the detected face
    return np.matrix([[pred.x, pred.y] for pred in predictor(image, face[0]).parts()])

#Drawing the landmarks : yellow in color
def landmarks_annotation(image, facial_landmarks):
    #Different image window for facial landmarks
    image = image.copy()
    for coord, p in enumerate(facial_landmarks):
        #Extracting coordinate values and the location / matrix of the coordinates
        position = (p[0, 0], p[0, 1])
        #Identify and draw the facial landmarks
        cv2.putText(image, str(coord), position, cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 255))
    return image

#Landmark coordinates for upper lip identified in the face 
def upperlip(facial_landmarks):
    ulip = []
    #create an array to store the landmark coordinates of the upper lip
    for i in range(50,53):
        #The range is predefined in "shape_predictor_68_face_landmarks.dat"
        ulip.append(facial_landmarks[i])
    for i in range(61,64):
        #The range is predefined in "shape_predictor_68_face_landmarks.dat"
        ulip.append(facial_landmarks[i])
    #Locate the mean value of the upper lip coordinates
    ulip_mean = np.mean(ulip, axis=0)
    return int(ulip_mean[:,1])#centroid value

#Landmark coordinates for lower lip identified in the face 
def lowerlip(facial_landmarks):
    llip = []
    #create an array to store the landmark coordinates of the lower lip
    for i in range(65,68):
        #The range is predefined in "shape_predictor_68_face_landmarks.dat"
        llip.append(facial_landmarks[i])
    for i in range(56,59):
        #The range is predefined in "shape_predictor_68_face_landmarks.dat"
        llip.append(facial_landmarks[i])
    #Locate the mean value of the lower lip coordinates
    llip_mean = np.mean(llip, axis=0)
    return int(llip_mean[:,1])#centroid value

#Detect the yawning activity
def yawning(image):
    #Obtain the facial Landmark coordinates
    facial_landmarks = get_facial_landmarks(image)
    if type(facial_landmarks) == str:
        return image, 0
    #Obtain the frame / image with annotated facial landmarks
    landmarks_image = landmarks_annotation(image, facial_landmarks)
    #Obtain Lip centroids
    upperlip_centroid = upperlip(facial_landmarks)
    lower_lip_centroid = lowerlip(facial_landmarks)
    #Calculate the distance between the centroids
    lips_dist = abs(upperlip_centroid - lower_lip_centroid)
    return landmarks_image, lips_dist
yawn_status = False 
yawn_count = 0

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
video_capture = cv2.VideoCapture("WIN_20220725_23_20_06_Pro.mp4")

# Check if camera opened successfully
if (video_capture.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
while(video_capture.isOpened()):
  # capture frame-by-frame
  ret, image_frame = video_capture.read()
  if ret == True:
    landmarks_image, lips_dist = yawning(image_frame)
    previous_status = yawn_status
    if lips_dist > 47:
        yawn_status = True
    else:
        yawn_status = False 
         
    if previous_status == True and yawn_status == False:
        yawn_count += 1

    # Display the resulting frame
    cv2.imshow('Frame',image_frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video video_captureture object
video_capture.release()

# Closes all the frames
cv2.destroyAllWindows()