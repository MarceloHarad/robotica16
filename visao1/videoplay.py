import cv2
import numpy as np
import math
import time



# O codigo está errado no momento de achar o centro do corredor. Fiz 
# o melhor que puder mas não consegui. No entando, consegui filtrar as 
# linhas para conseguir quase somente as bordas do corredor.



cap = cv2.VideoCapture('video_corredor.mp4') 

x1_final = [0]*2
x2_final = [0]*2
y1_final = [0]*2
y2_final = [0]*2


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(frame, 130, 300)
    cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    lines = cv2.HoughLinesP(edges, 1, math.pi/180.0, 40, np.array([]), 80, 50)



    if lines != None:
        for x1,y1,x2,y2 in lines[0]:
            if (x2 - x1) != 0:
                theta = (y2 - y1)/(x2 - x1)
                theta = np.arctan(theta)
            if theta != 0:
                if (theta < -0.7 and theta > -0.8) or (theta > 0.7 and theta <0.8):
                    if theta > 0:

                        x1_final[0] = x1
                        x2_final[0] = x2
                        y1_final[0] = y1
                        y2_final[0] = y2

                    if theta < 0:


                        x1_final[1] = x1
                        x2_final[1] = x2
                        y1_final[1] = y1
                        y2_final[1] = y2
                        
                    cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)

            # Display the resulting frame
    if x1_final[1] != 0:
        cv2.line(frame,(350,500),(sum(x2_final)/2,sum(y2_final)/2),(255,0,0),2)
    cv2.imshow('frame', frame)
    time.sleep(0.03)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()