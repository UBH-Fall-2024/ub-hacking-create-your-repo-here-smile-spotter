import cv2
import face_recognition
import numpy
from datetime import datetime
import sqlite3
import os
import mss
import threading

frame_count = 0
scale_factor = 1

#path to store faces in a seperate file on the desktop,
#makes a new folder if it doesn't exist
desktop_path = os.path.expanduser("~/Desktop")
output_folder = os.path.join(desktop_path,"detected_faces")
os.makedirs(output_folder, exist_ok=True)
#path to store profiles in the folder
profiles_folder = os.path.join(output_folder,"profiles")

def facesaver(face_image,profile_name):
    profile_folder = os.path.join(profiles_folder,profile_name)
    os.makedirs(profile_folder, exist_ok=True)
    #create a new filename and add a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_folder}/face_{timestamp}_{i}.jpg"
    #save the face into the profile photo
    cv2.imwrite(filename, face_image)
    print(f"Saved face to {profile_name}: {filename}")



def facializer(rgb_img, img, output_folder):
    small_img = cv2.resize(rgb_img, (0, 0), fx=scale_factor, fy=scale_factor)
    face_locations = face_recognition.face_locations(small_img)
    face_locations = [(int(top / scale_factor), int(right / scale_factor), int(bottom / scale_factor), int(left / scale_factor)) for top, right, bottom, left in face_locations]
    for i, (top, right, bottom, left) in enumerate(face_locations):
        #extract face image
        face_image = img[top:bottom, left:right]
        #save the image with a timestamp & unique name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_folder}/face_{timestamp}_{i}.jpg"
        cv2.imwrite(filename, face_image)
        print(f"Saved face image: {filename}")

        #draw rectangle around face and display their name
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0),2)
        cv2.putText(img, profile_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


#opens a screen capturing session and assigns the screen capture to the variable sct
with mss.mss() as sct:
    monitor = {"top":100,"left":100,"width":1800,"height":1169}
    while True:
        # Get raw pixels from the screen, save it to a Numpy array
        img = numpy.array(sct.grab(monitor))
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        if frame_count % 10 == 0:
            facializer(rgb_img,img,output_folder)
            #start a new thread for face detection & image saving
            #threading.Thread(target=facializer, args=(rgb_img, img, output_folder)).start()
        # Display the picture
        cv2.imshow("Facial Detection",img)

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
        frame_count+=1
        
#release the resources and close windows
cv2.destroyAllWindows()



#create the database where faces will be stored

#check if the face was smiling
#process each face
#look up each face
#display each face after processing with information