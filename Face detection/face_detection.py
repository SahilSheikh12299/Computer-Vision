import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv.VideoCapture(0)

while True:
    success, img = cap.read()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)  #Scalefactor = 1.1

    """
    Scale factor is the parameter which specifies how much image size is 
    reduced at each image scale
    """

    for (x,y,w,h) in faces:
        cv.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 3)
    
    cv.imshow('Img', img)
    
    #Quit if d is pressed on the keyboard
    if cv.waitKey(1) & 0xff == ord('d'):
        break
