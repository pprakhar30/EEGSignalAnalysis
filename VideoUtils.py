import cv2
import math
import os
from scipy import misc
import numpy as np
source = "/home/prakhar/SubjectBrainRecording"
dict1 = {}
for filename in os.listdir(source):
	cap = cv2.VideoCapture(source+"/"+filename)
	directory = filename
	os.makedirs(directory)
	os.chdir("./"+directory)
	number_frames = cap.get(7)
	total_frames = math.floor(number_frames/25)
	frames_remove = number_frames-total_frames*25
	#print filename+" :",12.0/cap.get(5)
	#print number_frames,cap.get(5),(number_frames*1.0/cap.get(5))
	count = 0
	while(True):
		ret,frame = cap.read()
		if not ret:
			break
		if frames_remove>0:
			frames_remove-=1
		else:
			resize_img = cv2.resize(frame, (64,64))
			cv2.imwrite("frame%d.jpg" % count, resize_img)
			count+=1
		
	os.chdir("..")
	print filename+" Done",total_frames
	dict1[filename] = total_frames
	cap.release()
	cv2.destroyAllWindows()

'''os.chdir("./Prakhar.mp4")
img = cv2.imread("frame0.jpg")
print img.shape,type(img)'''
print os.getcwd()
for filename in os.listdir(source):
	os.chdir("./"+filename)
	os.makedirs("TimedVersion")
	total_img_trial = dict1[filename]
	for i in xrange(25):
		base1 = i*total_img_trial
		for j in xrange(7):
			base2 = j*12
			accumulator = np.zeros((200,200,3))
			p = 12
			if j==6:
				if total_img_trial == 85:
					p = 13
				elif total_img_trial == 86:
					p = 14
			for k in xrange(p):
				base3 = k
				actual_addr = int(base1 + base2 + base3)
				#print os.getcwd(),actual_addr
				name_img = "frame" + str(actual_addr) +".jpg"
				img = cv2.imread(name_img)
				#print type(img),type(accumulator),img.shape,accumulator.shape
				accumulator+=img
			accumulator/=p
			newname = "Word_"+str(i)+"Seq_"+str(j)+".jpg"
			os.chdir("./TimedVersion")
			cv2.imwrite(newname,accumulator)
			os.chdir("..")
		print "Image Accumlation for "+str(i)+" word of  "+filename+" done" 
	os.chdir("..")





