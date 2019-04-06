import cv2
import numpy as np
import pyautogui
import math
import time
import threading

pyautogui.FAILSAFE = False
pos = None		# mouse location

# size of current screen
sW,sH = pyautogui.size()

# window variants
wW,wH = sW//2, sH//2
wX,wY = sW-wW, sH-wH

window = 'Mouse Control'
cv2.namedWindow(window,cv2.WINDOW_AUTOSIZE)
cv2.moveWindow(window,wX,wY)

# size of frame
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, wW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, wH)

# region of interest
bX,bY,bW,bH = wW//2,0,wW//2,wH//2

# background subtractor model
fgbg = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

def calculateDistance(p1,p2):
	return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# check angle between 3 points in convex hull
# if angle < 90, we assume that this is a finger
def checkAngle(start,end,far):
	a = calculateDistance(start,end)
	b = calculateDistance(far,start)
	c = calculateDistance(far,end)
	angle = math.acos((b**2+c**2-a**2)/(2*b*c))

	return angle < math.pi/2

# check distance from start point to center of countour
# if distance > S(contour)*0.2, we assume that this is a finger
def checkSize(start,center,w):
	return calculateDistance(start,center) > 0.2*w

# check coordination of start point
# if x-coord and y-coord of start point lower than center
# we assume that this is not a finger
def checkCoordinate(start,center):
	return start[0] > center[0] and start[1] > center[1]

def calculateFingers(cnt,frame):
	global pos
	if cv2.contourArea(cnt) > 2000:
		epsilon = 0.01*cv2.arcLength(cnt,True)
		approx = cv2.approxPolyDP(cnt,epsilon,True)
		x,y,w,h = cv2.boundingRect(cnt)

		hull = cv2.convexHull(cnt,returnPoints=False)
		M = cv2.moments(cnt)
		cX = int(M['m10']/M['m00'])
		cY = int(M['m01']/M['m00'])
		center = (cX,cY)

		defects = cv2.convexityDefects(cnt,hull)

		if defects is not None:
			fingers = 1
			for i in range(defects.shape[0]):
				s,e,f,d = defects[i][0]
				start = tuple(cnt[s][0])
				end = tuple(cnt[e][0])
				far = tuple(cnt[f][0])

				if checkAngle(start,end,far) and checkSize(start,center,w) and not checkCoordinate(far,center):
					fingers += 1
			
			if fingers == 1 or fingers == 2:
				# finding extreme top point of contour, our fingertip
				extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
				pos = extTop
				cv2.circle(roi,extTop,8,(0,0,255),-1)
			
			return fingers
	else:
		pos = None	
		return 0

# map finger tip location with mouse on screen
def move():
	while(True):
		if pos is None:
			time.sleep(pyautogui.MINIMUM_SLEEP)
		else:
			mX,mY = int((pos[0])*sW/bW), int((pos[1])*sH/bH)
			# this help mouse can reach to screen angle
			# as webcam hardly detect our fingertip when we move hand to far location
			if pos[0] > bX/2:
				mX += 200
				mY -= 100
			pyautogui.moveTo(mX,mY,duration=pyautogui.MINIMUM_DURATION)

thread = threading.Thread(target=move)
thread.daemon = True
thread.start()

actions = []

def get_action(actions):
	action = max(actions,key=actions.count)
	if actions.count(action)/len(actions) > 0.9:
		return action
	return None

def handle(action):
	if action is not None:
		if action == 4:
			pos = None
			pyautogui.click(interval=0.1)
		else:
			pass

while True:
	ok,frame = cap.read()
	frame = cv2.flip(frame,1)
	# draw region of interest on frame
	cv2.rectangle(frame,(bX,bY),(bX+bW,bY+bH),(0,255,0),2)
	
	# create a mask for this region of interest
	roi = frame[bY:bY+bH,bX:bX+bW]
	blur = cv2.GaussianBlur(roi,(11,11),0)
	mask = fgbg.apply(blur,learningRate=-1)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

	cnts,hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	if len(cnts) > 0:
		cnt = max(cnts,key=cv2.contourArea)
	
	cv2.drawContours(roi,[cnt],0,(255,255,0),2)
	actions.append(calculateFingers(cnt,roi))
	
	if len(actions) > 20:
		actions = actions[-20:]
		action = get_action(actions)

		handle(action)

	cv2.imshow(window,frame)
	k = cv2.waitKey(1) & 0xff

	if k == ord('q'):
		break
	
cap.release()
cv2.destroyAllWindows()