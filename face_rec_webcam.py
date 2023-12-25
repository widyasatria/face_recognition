#Source : Live Face Recognition
#https://www.youtube.com/watch?v=sz25xxF_AVE

import cv2
import face_recognition
import numpy as np
import os, re
import msvcrt
import math

def findEncodings(images):
    encodeList =[]
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
        


def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))


def main():
    
    path="images"
    tolerance_face_comparison = 0.48 #default 0.6
    images = []
    classNames = []
    myList = os.listdir(path)
    #print(myList) #printing semua file yang ada untuk disimpan kedalam list

    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)           # develop image list
        fname = os.path.splitext(cl)[0] # removing the extension
        classNames.append(re.sub(r'\d+','',fname)) #using regex removing the number, keeping classnames/name only

    
    encodeListKnown= findEncodings(images)
    print("Encoding {} images completed".format(len(encodeListKnown)))
    
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        imgS = cv2.resize(img,(0,0),None,0.25,0.25) #resize image to make the recognition faster
        imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS) #detect all face in the current framew
        encodeCurFrame = face_recognition.face_encodings(imgS,facesCurFrame) #encode all faces in the current framew
        
        for encodeFace, faceLoc in zip(encodeCurFrame,facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown,encodeFace, tolerance_face_comparison=0.48) #compare encoded face in current framew with existing known encode listknown add tolerance 0.l51
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace) #lowest distance will be the best matche
            
            #print("all registered classmates {} ".format(classNames))
            matchIndex = np.argmin(faceDis) # to get the index of matching image
            print(" name : {} and face distance  : {}".format(classNames[matchIndex],faceDis[matchIndex]))
            
            #print(faceLoc)
            y1, x2, y2, x1 = faceLoc[0]*4,faceLoc[1]*4,faceLoc[2]*4,faceLoc[3]*4
            distance_percent = round(face_distance_to_conf(faceDis[matchIndex],tolerance_face_comparison),2) * 100 #untuk convert face_distance menjadi percentage
            
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                cv2.putText(img,name,(x1,y1-40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2) 
                cv2.putText(img,"Akurasi : "+str(distance_percent)+"%",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2) 
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2,cv2.FILLED)
                #print(" isi dari nama {} x1 {} y1 {} ".format(name,x1, y1) )
                
            else:
                no_match = "NOT MATCH"
                text = "Akurasi : " + str(distance_percent) + "%"
                cv2.putText(img,no_match,(x1,y1-35),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,200),2) 
                cv2.putText(img,text,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,200),2) 
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,200),2)
           
            #print("Matches[matchIndex] {}".format(matches[matchIndex]))
            cv2.imshow("Frame",img)
            
            key = cv2.waitKey(1)
        if msvcrt.kbhit() and msvcrt.getch() == chr(27).encode():
            break
         
if __name__ == "__main__":
    main()
    
     


