import numpy as np
import cv2

#Foto base
img1 = cv2.imread('distance_image.png')  

cv2.imshow("Foto", img1)

cv2.waitKey(0)

dst = cv2.Canny(img1, 50, 200)
cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

#Video
cap = cv2.VideoCapture('garrafa.mp4')

HEIGHT = int(cap.get(4))
WIDTH = int(cap.get(3))

# Initiate SIFT detector
sift = cv2.SIFT()

kp1, des1 = sift.detectAndCompute(img1,None)

MIN_MATCH_COUNT = 10


#Distancia Inicial da Garrafa na foto - 45 cm
#Tamanho da garrafa medido - 30 cm
#Tamanho da garrafa em pixels - 832

#Portanto, distancia focal em pixels - 1248
#Assim, distancia final em pixels (Distancia focal X Tamanho real) - 37440 pixels

distancia_pixels = 37440

while True: 

	ret, frame = cap.read()

	if frame != None:

		kp2, des2 = sift.detectAndCompute(frame,None)


		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks = 50)

		flann = cv2.FlannBasedMatcher(index_params, search_params)

		matches = flann.knnMatch(des1,des2,k=2)

		# store all the good matches as per Lowe's ratio test.
		good = []	

		for m,n in matches:
		    if m.distance < 0.7*n.distance:
		        good.append(m)

		if len(good)>MIN_MATCH_COUNT:
			src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
			dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

			M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
			matchesMask = mask.ravel().tolist()

			h,w = img1.shape[0], img1.shape[1]
			pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

			dst = cv2.perspectiveTransform(pts,M)

			aux1 = abs(dst[1][0][0] + dst[0][0][0])
			aux2 = abs(dst[2][0][0] + dst[3][0][0])
			altura_pixel = (aux2 - aux1)

			D = distancia_pixels / altura_pixel

			#Verificar se esta em um range aceitavel para ter identificado a garrafa
			if D < 150 and D > 0:
				img2b = cv2.polylines(frame,[np.int32(dst)],True,255,3, cv2.CV_AA)
				cv2.putText(frame,str(int(D)) + " cm",(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.CV_AA)
		else:
			print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
			matchesMask = None


		cv2.imshow('final',frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	else:
		print "No frame found"  

cv2.destroyAllWindows()