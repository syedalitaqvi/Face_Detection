import cv2
import numpy as np

#classifier object
face_classifier = cv2.CascadeClassifier('C:/Users/user/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

#Function to extract features
def face_extractor(img):
    
    #converting RGB2 to GREY_SCALE
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #Calling function through classifier
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    #if no face is there
    if faces is():
        return None

    #if face is there
    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+h]

    return cropped_face
    
#To configure camera
cap = cv2.VideoCapture(0)
count=0

while True:
    ret,frame = cap.read()
    if face_extractor(frame) is not None:
        count+=1
        #Resizing the face
        face = cv2.resize(face_extractor(frame),(200,200))

        #Converting to grayscale
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        #Now saving face values
        file_name_path = 'H:/DATASET/user'+str(count)+'.jpg'
        cv2.imwrite(file_name_path,face)

        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)
         
    else:
        print("Face Not Found")
        pass

    if cv2.waitKey(1)==13 or count==100:
        break
#Camera is closed   
cap.release()
cv2.destroyAllWindows()
print('Collecting of samples is complete!!')
