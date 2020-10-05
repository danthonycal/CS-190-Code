import numpy as np
import cv2
import time 
import sys
import glob

color=(255,0,0)
thickness=2

tracker_types = ['KCF', 'MOSSE', 'CSRT']
mode = sys.argv[1]
vid_index = sys.argv[2]
videos = glob.glob("./videos/*.mp4")
tracker_type = int(mode)
print('TRACKER: ' + tracker_types[tracker_type])
print('VIDEO: ' + videos[int(vid_index)])

cap = cv2.VideoCapture(videos[int(vid_index)])
# cap = cv2.VideoCapture(0)

ok, first_frame = cap.read()

while(True):
    # Capture two frames
    ret, frame1 = cap.read()  # first image
    time.sleep(1/25)          # slight delay
    ret, frame2 = cap.read()  # second image 
    img1 = cv2.absdiff(frame1, frame2)  # image difference
    
    # get theshold image
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img_ = gray
    
    img_ =  cv2.resize(img_,(640, 480))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img_ = cv2.morphologyEx(img_, cv2.MORPH_CLOSE, kernel)
    
    img_ = cv2.dilate(img_, kernel, iterations = 2)
    img_ = cv2.GaussianBlur(img_, (1, 1), 0)
    img_ = cv2.morphologyEx(img_, cv2.MORPH_OPEN, kernel)
    

    ret,thresh = cv2.threshold(img_, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    cv2.imshow('thresh', thresh)

    # combine frame and the image difference
    img2 = cv2.addWeighted(frame1, 0.99, img1, 0.01, 0)
    img2 = cv2.resize(img2, (640, 480), interpolation = cv2.INTER_AREA)
    # get contours and set bounding box from contours
    img3 = img2
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        for c in contours:
            if tracker_type == 0:
                tracker = cv2.TrackerKCF_create()
            if tracker_type == 1:
                tracker = cv2.TrackerMOSSE_create()
            if tracker_type == 2:
                tracker = cv2.TrackerCSRT_create()
            rect = cv2.boundingRect(c)
            bbox = rect
            height, width = img3.shape[:2]            
            if rect[2] > 0.2*height and rect[2] < 0.7*height and rect[3] > 0.2*width and rect[3] < 0.7*width: 
                x,y,w,h = cv2.boundingRect(c)            # get bounding box of largest contour
                ok = tracker.init(img2, (x, y, w, h))
                img4=cv2.drawContours(img2, c, -1, color, thickness)
                img5 = cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 0, 255), 2)  # draw blue bounding box in img
                ok, bbox = tracker.update(img2)
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(img2, p1, p2, (255, 0, 0), 2, 1)
                if ok:
                    # Tracking success
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(img2, p1, p2, (255, 0, 0), 2, 1)
                else :
                    # Tracking failure
                    cv2.putText(img2, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0, 0, 255), 2)
            else:
                img5=img2
    else:
        img5=img2
        
    # Display the resulting image
    
    cv2.imshow('Motion Detection by Image Difference', img2)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
        break
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()