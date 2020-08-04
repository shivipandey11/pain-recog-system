import cv2
import sys
import numpy as np
from model import EMR

# prevents opencl usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

EMOTIONS = ['angry', 'disgusted', 'fearful', 'smile', 'sad', 'surprised', 'neutral']

def format_image(image):
    """
    Function to format frame
    """
    if len(image.shape) > 2 and image.shape[2] == 3:
        # determine whether the image is color
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # Image read from buffer
        image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = cascade_classifier.detectMultiScale(image,scaleFactor = 1.3 ,minNeighbors = 5)

    if not len(faces) > 0:
        return None

    # initialize the first face as having maximum area, then find the one with max_area
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    face = max_area_face

    # extract ROI of face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]

    try:
        # resize the image so that it can be passed to the neural network
        image = cv2.resize(image, (48,48), interpolation = cv2.INTER_CUBIC) / 255.
    except Exception:
        print("----->Problem during resize")
        return None

    return image

# Initialize object of EMR class
network = EMR()
network.build_network()

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
feelings_faces = []

# append the list with the emoji images
for index, emotion in enumerate(EMOTIONS):
    feelings_faces.append(cv2.imread('./emojis/' + emotion + '.png', -1))

while True:
    # Again find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
    
    
    # compute softmax probabilities
    result = network.predict(format_image(frame))
    if result is not None:
        #result[0][6]=result[0][6]*5/10
        # write the different emotions and have a bar to indicate probabilities for each class
        if result[0][5]>0.5:
            pain=0
        else:
            pain=3*result[0][4]+3*result[0][0]+1*result[0][1]+4*result[0][2]+1.5*result[0][3]+0.5*result[0][5]-0.05*result[0][6]
        for index, emotion in enumerate(EMOTIONS):
                cv2.putText(frame, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1);
                cv2.rectangle(frame, (130, index * 20 + 10), (130 + int(result[0][index] * 100), (index + 1) * 20 + 4), (255, 0, 0), -1)
            # find the emotion with maximum probability and display it
                #maxindex = np.argmax(result[0])
                #font = cv2.FONT_HERSHEY_SIMPLEX
                #cv2.putText(frame,EMOTIONS[maxindex],(10,200), font, 2,(255,255,255),2,cv2.LINE_AA) 
                #face_image = feelings_faces[maxindex]
            
                #painlevel of the person along with the bar
                if(pain<1.35):
                    cv2.putText(frame,"Pain level:"+str(pain),(10,350), font, 1,(255,255,255),2,cv2.LINE_AA)
                elif(pain>=1.35 and pain<2.1):
                    cv2.putText(frame,"Pain level:"+str(pain),(10,350), font, 1,(0,255,255),2,cv2.LINE_AA)
                else:
                    cv2.putText(frame,"Pain level:"+str(pain),(10,350), font, 1,(0,0,255),2,cv2.LINE_AA)
                cv2.rectangle(frame, (115, 400), (115+ int(pain*90), 380), (255, 0, 0), -1)
            
                #for c in range(0, 3):
                    # The shape of face_image is (x,y,4). The fourth channel is 0 or 1. In most cases it is 0, so, we assign the roi to the emoji.
                    # You could also do: frame[200:320,10:130,c] = frame[200:320, 10:130, c] * (1.0 - face_image[:, :, 3] / 255.0)
                    #frame[200:320, 10:130, c] = face_image[:,:,c]*(face_image[:, :, 3] / 255.0) +  frame[200:320, 10:130, c] * (1.0 - face_image[:, :, 3] / 255.0)
            
            

    if len(faces) > 0:
        # draw box around face with maximum area
        max_area_face = faces[0]
        for face in faces:
            if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
                max_area_face = face
        face = max_area_face
        (x,y,w,h) = max_area_face
        frame = cv2.rectangle(frame,(x,y-50),(x+w,y+h+10),(255,0,0),2)

        cv2.imshow('Video', cv2.resize(frame,None,fx=2,fy=2,interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

#pain = 6*result[0][4]+4*result[0][0]+2*result[0][1]+result[0][2]-4*result[0][3]+2*result[0][5]
