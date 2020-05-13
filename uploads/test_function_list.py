# 4/16 8:07pm JD - added has arrived attributes to keep track of when a tracked person arrives at the predicted camera


class ActivityDbRow(object):
    def __init__(self, row=None):
        # these values correspond with the columns in the database
        self.id = None
        self.label = None
        self.start_time = None
        self.end_time = None
        self.camera_id = None
        self.next_camera_id = None
        self.has_arrived = None

        # these values are used at runtime to track various aspects of the tracked person
        self.rect_start = None  # the x,y upper left coordinate of the bounding rect
        self.rect_end = None  # the x, y of the lower right ...
        self.detected = False  # this value gets marked as true on each pass through the main camera loop until the person leaves and then becomes false
        # an additional value used to decide when a person has left the camera
        self.not_detected_count = 0
        self.updateLabelCounter = 0
        if row:  # when reconsituting from the database we will have a row of column values that we use to populate this instance
            self.id = row[0]
            self.label = row[1]
            self.start_time = row[2]
            self.end_time = row[3]
            self.camera_id = row[4]
            self.next_camera_id = row[5]
            self.has_arrived = True if row[6] and row[6] == 'T' else False

# below are general setter and getter methods for the above attributes
    def getID(self):
        return self.id

    def setID(self, id):
        self.id = id

    def getLabel(self):
        return self.label

    def setLabel(self, label):
        if self.label == None or self.label == "Unknown":
            self.updateLabelCounter == 0
            self.label = label
        elif self.updateLabelCounter == 5:
            self.updateLabelCounter == 0
            if label != "Unknown":
                self.label = label
        else:
            self.updateLabelCounter += 1

    def getStart_time(self):
        return self.start_time

    def setStart_time(self, start_time):
        self.start_time = start_time

    def getEnd_time(self):
        return self.end_time

    def setEnd_time(self, end_time):
        self.end_time = end_time

    def getCamera_id(self):
        return self.camera_id

    def setCamera_id(self, camera_id):
        self.camera_id = camera_id

    def getNext_camera_id(self):
        return self.next_camera_id

    def setNext_camera_id(self, next_camera_id):
        self.next_camera_id = next_camera_id

    def get_has_arrived(self):
        return self.has_arrived

    def set_has_arrived(self, b):
        self.has_arrived = b

    def getRect_start(self):
        return self.rect_start

    def setRect_start(self, point):
        self.rect_start = point

    def getRect_end(self):
        return self.rect_end

    def setRect_end(self, point):
        self.rect_end = point

    def set_detected(self, b):
        if b:
            self.not_detected_count = 0
        self.detected = b

    def was_detected(self):
        return self.detected

    # only if this gets called 5 times does it finally return true
    # it insures that we have indeed encountered an activity that
    # has left the camera but we don't know which way they went
    # five times = a half a second of time
    def has_left_the_scene(self):
        self.not_detected_count += 1
        return self.not_detected_count > 5

    # some basic sql methods for common operations on an activitydbrow
    def getSelectStatement(self):
        return "select id, label, start_time, end_time, camera_id, next_camera_id, has_arrived from tracking where id = %s" % self.id

    # when updating a tracking record we are only updating the end_time, next_camera_id and has_arrived columns
    def getUpdateStatement(self):
        return "update tracking set end_time = current_timestamp, next_camera_id = %s, has_arrived = '%s' where id = %s" % ((self.next_camera_id if self.next_camera_id else 'null'), 'T' if self.has_arrived else 'F', self.id)

    # when inserting we are populating the lable, camera_id, raw_time and has_arrived columns ( the database uses an auto increment id field that assigns the id )
    def getInsertStatement(self):
        return "insert into tracking (label, camera_id, raw_time, has_arrived) values('%s', %s, '%s', 'F')" % (self.label, (self.camera_id if self.camera_id else 'null'), self.start_time)
# camera.py
# 4/10 9:51pm LH - person tracking enhancement and some comment
# 4/13 8:49pm JL - modified label assgignment logic to reuse original label for the same person at new camera.
# 4/16 8:07pm JD - better detection of when someone leaves view and more accurate label reuse
# 4/17 9:00pm LH,JS - Fixed the get_label query to update the has_arrived correctly
# 4/18 7:46pm SH,JL - add better matching logic when additional people come into view of a camera
import os
import time
import face_recognition
from shared.ActivityDbRow import ActivityDbRow
from shared.CameraDbRow import CameraDbRow
from threading import Lock
import imutils
import numpy as np
import datetime
import cv2
import sys
sys.path.append("..")

port = 5001
if 'PORT' in os.environ:
    port = int(os.environ['PORT'])


def whichHalf(x):
    if x < 128:
        return 0
    else:
        return 1

# used as part of the prediction algorithm and also to help facility keeping the same person labeled correctly


def distance(p1, p2):
    # calculates the distance between two points
    return ((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)**0.5

# an instance of this class manages the camera hardware


class VideoCamera(object):
    def __init__(self, cv2_index, cameraDetails, mysql):
        # Using OpenCV to capture from device identified by cv2_index.  Some laptops have a built in camera in addition
        # to the usb camera we are using and these cameras are assigned an integer value starting with 0
        # the db info about this particular camera - see CameraDbRow for more info
        self.cameraDetails = cameraDetails
        self.mysql = mysql  # mysql db reference
        self.shutItDown = False  # as long as this flag is false the camer will keep running
        # a cv2 specific class for talking to camera hardware
        self.camera = cv2.VideoCapture(int(cv2_index))
        self.net = cv2.dnn.readNetFromCaffe(
            "MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")
        # initialize the list of class labels MobileNet SSD was trained to
        # detect, then generate a set of bounding box colors for each class

        ret, self.no_video = cv2.imencode('.jpg', cv2.imread(os.path.realpath(
            "./no_video.jpg")))  # set the no_video image to display when the camer is off
        self.jpeg = self.no_video
        # the initial state is that no capturing is happening until the start method is activated
        self.capturing = False
        self.lock = Lock()  # a lock used when allowing access to the video feed by the browser
        # the list of currently tracked activities ( simultaneous people refrences )
        self.tracked_list = []
        self.used_activity = []  # of the activities being tracked, on each frame this list keeps track of the activities that are still active, all other activities represent people who have left
        # this keeps track of the person that most recently left and is used to detect if they happen to return again to the same camera
        self.recently_left = None

    def __del__(self):
        self.camera.release()

    # given a activity id this method can load the corresponding row from the database into an ActivityDbRow instance
    # used when we are trying to determine if we are seeing the same recently left person return
    def loadActivityDb(self, id):
        a = ActivityDbRow()
        a.setID(id)
        cursor = self.mysql.connect().cursor()
        cursor.execute(a.getSelectStatement())
        data = cursor.fetchone()
        if data:
            a = ActivityDbRow(data)
        return a

    # insert a new activity in the tracking table
    # after the insert we must select the assigned id back into the activity record for future use
    def insertActivity(self, activity):
        conn = self.mysql.connect()
        cursor = conn.cursor()
        cursor.execute(activity.getInsertStatement())
        conn.commit()
        cursor = self.mysql.connect().cursor()
        # raw_time field is an alternate key that allows us to find the newly inserted row and get it's id
        sql = "select id from tracking where raw_time = '%s' and camera_id = %s" % (
            activity.getStart_time(), activity.getCamera_id())
        cursor.execute(sql)
        data = cursor.fetchone()
        if data:
            activity.setID(data[0])

    # update a preexisting activity in the tracking table
    def saveActivity(self, activity):
        if activity.getID():
            conn = self.mysql.connect()
            cursor = conn.cursor()
            cursor.execute(activity.getUpdateStatement())
            conn.commit()

    def saveRecoveredActivity(self, activity):
        if activity.getID():
            conn = self.mysql.connect()
            cursor = conn.cursor()
            cursor.execute(
                "update tracking set end_time = null, next_camera_id = null, has_arrived = 'F' where id = %s" % activity.getID())
            conn.commit()

    # We use this to interact with the neural net data returned form cv2 to build up the list of starting rectangle coordinates for all detected people
    # this method is called up front to know ahead of the detection logic how many people we are dealing with
    def get_all_detected_points(self, detections, h, w):
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        points = []
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                idx = int(detections[0, 0, i, 1])
                # at this point we know we are dealing with a person ( see similar logic below with comments )
                if confidence > 0.5 and (idx == 15):
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    points.append(box.astype("int")[0:2])

        return points

    # look for a face inside the rectangle bounding a person.
    # using the location of the face, find a smaller region
    # rougly where the chest should be to detect shirt color
    # def identify(self, sub_frame, cv2):
    # 	BLUE=(255, 0, 0)
    # 	SHIRT_DY = 1.75;	# Distance from top of face to top of shirt region, based on detected face height.
    # 	SHIRT_SCALE_X = 0.6;	# Width of shirt region compared to the detected face
    # 	SHIRT_SCALE_Y = 0.6;	# Height of shirt region compared to the detected face
    # 	label = None
    # 	try:
    # 		gray = cv2.cvtColor(sub_frame, cv2.COLOR_BGR2GRAY)
    # 		gray = cv2.GaussianBlur(gray, (21, 21), 0)
    # 		face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    # 		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # 		for (x,y,w,h) in faces:
    # 			x = x + int(0.5 * (1.0-SHIRT_SCALE_X) * w);
    # 			y = y + int(SHIRT_DY * h) + int(0.5 * (1.0-SHIRT_SCALE_Y) * h);
    # 			w = int(SHIRT_SCALE_X * w);
    # 			h = int(SHIRT_SCALE_Y * h);
    # 			cv2.rectangle(sub_frame, (x, y), (x+w, y+h), BLUE, 1)
    # 			label = "Person %s" % self.getIdentitiyCode(sub_frame[y:(y+h),x:(x+w)])
    # 			print(label)
    # 	except Exception:
    # 		None
    # 	return label

    def identify(self, sub_frame, cv2):
        BLUE = (255, 0, 0)
        # Distance from top of face to top of shirt region, based on detected face height.
        SHIRT_DY = 1.75
        SHIRT_SCALE_X = 0.6  # Width of shirt region compared to the detected face
        SHIRT_SCALE_Y = 0.6  # Height of shirt region compared to the detected face
        label = None
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = sub_frame[:, :, ::-1]

        # Find all the faces and face enqcodings in the frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_frame, face_locations)
        # Loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            x = left
            y = top
            w = right-left
            h = bottom-top
            x = x + int(0.5 * (1.0-SHIRT_SCALE_X) * w)
            y = y + int(SHIRT_DY * h) + int(0.5 * (1.0-SHIRT_SCALE_Y) * h)
            w = int(SHIRT_SCALE_X * w)
            h = int(SHIRT_SCALE_Y * h)
            cv2.rectangle(sub_frame, (x, y), (x+w, y+h), BLUE, 1)
            label = "Person %s" % self.getIdentitiyCode(
                sub_frame[y:(y+h), x:(x+w)])

        return label

    def saveActivityLabel(self, t):
        conn = self.mysql.connect()
        cursor = conn.cursor()
        print(("saving %s", t.getLabel()))
        cursor.execute("update tracking set label = '%s' where id = %s" % (
            t.getLabel(), t.getID()))
        conn.commit()

    # given a subregion ( at chest level ) we calculate the average pixel color and then
    # use that to index down to a numeric value in the range of 1-6.
    def getIdentitiyCode(self, img):
        avg_color_per_row = np.average(img, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        (b, g, r) = avg_color
        print(("%s %s %s" % (r, g, b)))
        if r < 128 and b < 128 and g < 128:
            return 1
        elif r > 200 and b > 200 and g > 200:
            return 2
        elif r > b and r > g:
            return 3
        elif b > g and b > r:
            return 4
        elif g > b and g > r:
            return 5
        else:
            return 6

    # start contains the main camera loop and is called by our background thread - see main.py for how it gets called
    def start(self):
        GREEN = (0, 255, 0)  # a color value for drawing our green boxes
        BLUE = (0, 0, 255)
        # each loop is a frame of video - do we see people in this frame?
        while self.camera.isOpened():  # loop until the camer is closed
            self.used_activity = []  # initialize to an empty list on each frame
            if self.shutItDown:  # when this flag is true we shutdown camera and then the loop exits
                self.camera.release()
                break

            # indicate to the outside world that we are capturing a feed from the video hardware
            self.capturing = True
            # read a frame of video from cv2 camera instance
            (grabbed, frame) = self.camera.read()
            if not grabbed:  # if no frame is returned this will be false and we'll loop back to the top of the while loop
                continue
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels
            frame = imutils.resize(frame, width=400)

            # grab the frame dimensions and convert it to a blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                         0.007843, (300, 300), 127.5)

            # pass the blob through the network and obtain the detections and
            # predictions
            self.net.setInput(blob)
            detections = self.net.forward()
            # count how many people we are tracking up front here
            all_detected_points = self.get_all_detected_points(
                detections, h, w)
            # initialize detected value of the activities we are tracking to false up front and those that are
            # still false at the end of the loop are activities we may no longer be observiing
            for t in self.tracked_list:
                t.set_detected(False)

            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence > 0.2:
                    # extract the index of the class label from the
                    # `detections`, if it's 15 then we know it's a person
                    idx = int(detections[0, 0, i, 1])
                    # the rectangle bounding the person
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    # extract into variables
                    (startX, startY, endX, endY) = box.astype("int")
                    if confidence > 0.5 and (idx == 15):
                        # we've found a person with a confidence level greater than 50 percent
                        # rectangle start coordinate upper left
                        rect_start = (startX, startY)
                        # rectangle end coordinate - lower right
                        rect_end = (endX, endY)

                        # we use this function call to associate the bounding box we are working on
                        # right now with the closest activity from the previous frame
                        # if no previous activities are being tracked then a new activity is created
                        newLabel = self.identify(
                            frame[startY:endY, startX:endX], cv2)
                        t = self.find_closest_tracked_activity(
                            rect_start, newLabel, all_detected_points)
                        # only use a label if we found one
                        if newLabel != None:
                            t.setLabel(newLabel)
                            self.saveActivityLabel(t)
                        # mark it as being detected so we know it's an active tracking
                        t.set_detected(True)
                        t.setRect_start(rect_start)
                        t.setRect_end(rect_end)
                        # draw the prediction on the frame
                        label = "{}: {:.2f}%".format(
                            t.getLabel(), confidence * 100)
                        cv2.rectangle(frame, rect_start, rect_end, GREEN, 2)

                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(frame, label, (startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)

            # performance enhancement to only grab a frame once per 100 milliseconds ( 10 frames per second )
            time.sleep(.1)

            # this logic tries to determine who left the camera
            removed_from_tracking = []
            # loop over everthing we are currently tracking
            for t in self.tracked_list:
                if not t.was_detected():  # we think this one is gone
                    # guard againest false positives 'person should not have been in view for only 2 seconds'
                    if time.time() - t.getStart_time() > 2:
                        # which way did they go?
                        if self.went_left(t):
                            print(("went left heading to %s" %
                                   self.cameraDetails.left_camera_id))
                            t.setNext_camera_id(
                                self.cameraDetails.left_camera_id)
                            removed_from_tracking.append(t)
                        elif self.went_right(t):
                            print(("went right heading to %s" %
                                   self.cameraDetails.right_camera_id))
                            t.setNext_camera_id(
                                self.cameraDetails.right_camera_id)
                            removed_from_tracking.append(t)
                elif t.has_left_the_scene():
                    # if someone leaves view and we don't detect it correctly, mark them arrived and remove from tracking
                    # the threshold for this is 5 times through the loop and we don't see them
                    t.set_has_arrived(True)
                    removed_from_tracking.append(t)

            # now we can remove all activities that are truly gone and save them in the db
            # this has to be done separate than the above loop because we can't modify the list
            # we are activly looping over
            for t in removed_from_tracking:
                # remove tracked entries from tacked_list that were in removed_from_tracking list
                self.saveActivity(t)
                t.setEnd_time(time.time())
                self.recently_left = t
                del self.tracked_list[self.tracked_list.index(t)]

            # update the jpeg that we serve back to clients
            self.lock.acquire()
            ret, self.jpeg = cv2.imencode('.jpg', frame)
            self.lock.release()

        # while loop has exited so we are no longer capturing video, set the jpeg to the no_video image
        self.capturing = False
        self.lock.acquire()
        self.jpeg = self.no_video
        self.lock.release()
    print('camera released.')

    # used to get an integer identifier for a new tracked person
    def get_next_person_number(self):
        conn = self.mysql.connect()
        cursor = conn.cursor()
        cursor.execute("select count(distinct label) from tracking")
        data = cursor.fetchone()
        if data:
            return int(data[0]) + 1

    # get the label to display and to store in the tracking table
    def get_label(self):
        conn = self.mysql.connect()
        cursor = conn.cursor()
        camera_id = self.cameraDetails.getID()
        l = "Unknown"
        # try to find the original label for this tracked person rather than creating a new label
        # the label for an activity record (for this camera) that indicates that someone is supposed to arrive but hasn't, needs to be used as the label if one is found
        # because we predicted someone would arrive and now someone has, we are assuming it's the same person so reuse the label
        cursor.execute(
            "SELECT id, label from tracking where next_camera_id is not null and next_camera_id = %s and has_arrived = 'F' order by start_time asc limit 1" % (camera_id))
        data = cursor.fetchone()
        if data:
            previous_id = data[0]
            # use this label instead of the one we were going to use
            l = data[1]
            # update the prediction logic so that the yellow indicator turns off at the same time the motion indicator turns on at this camera
            if previous_id:
                conn.cursor().execute("update tracking set has_arrived = 'T' where id = %d" % previous_id)
                conn.commit()
        return l

    # method to find a tracking activity record that corresponds with the person detected in this frame represented by rect_start and newLabel
    def find_closest_tracked_activity(self, rect_start, newLabel, all_detected_points):
        # populate a variable with the number of detected people in frame at this time
        detected_person_count = len(all_detected_points)
        # remove rect_start from all_detected_points - any points in the list that are not rect_start are kept by this lambda expression
        all_detected_points_except_this_one = list(
            [x for x in all_detected_points if x[0] != rect_start[0] or x[1] != rect_start[1]])
        # find all the traced activities not yet paired up with a person in this frame
        self.unused_tracked_list = list(
            set(self.tracked_list) - set(self.used_activity))
        # if list is empty then just add a new activity
        if not self.tracked_list:
            return self.begin_new_tracking(rect_start)
        else:
            # otherwise use the distance formula to find the tracked activity that is closest to this new point
            closest_t = None
            for t in self.unused_tracked_list:
                if closest_t:
                    # first find the next closest match
                    closest_t = t if distance(t.getRect_start(), rect_start) < distance(
                        closest_t.getRect_start(), rect_start) else closest_t
                    # use if the labels match.  This keeps a "swap" from happening when multiple people are close together
                    if newLabel != None and closest_t.getLabel() == newLabel:
                        self.used_activity.append(closest_t)
                        return closest_t  # just return this one because it must be the match
                else:
                    closest_t = t

            # we might not want to use this one if it's closer to someone else
            # and we are tracking more than one person
            more_people_than_activities = detected_person_count > len(
                self.tracked_list)
            # if the activity found above is actually closer to one of the other people in frame, then don't pair it to this person, instead create a new one
            if not closest_t or (more_people_than_activities and self.is_this_activity_closer_to_someone_else(closest_t, all_detected_points_except_this_one, rect_start)):
                print(more_people_than_activities)
                print(closest_t)
                closest_t = self.begin_new_tracking(rect_start)

            # mark it as used here so that the next pass through the detection loop above, we don't try to use it again
            self.used_activity.append(closest_t)
            return closest_t

    # search through the list "the_others" and find any matches that are closer to "activity" than "me"
    def is_this_activity_closer_to_someone_else(self, activity, the_others, me):
        # closeness is determined by using the upper left rectangle coordinates and the distance formula
        activity_rect = activity.getRect_start()
        # find the distance between me and the activity that I am being matched with
        distance_to_me = distance(activity_rect, me)
        # use that distance to filter out other matches that are closer
        matches = list([x for x in the_others if distance(
            activity_rect, x) < distance_to_me])
        # if any were found, return true otherwise false
        return len(matches) > 0

    # begin a new ActivityDbRow instance to track a new person in frame
    def begin_new_tracking(self, rect_start):
        t = None
        # see if a recently leaving activity has returned
        if self.recently_left:
            d = distance(rect_start, self.recently_left.getRect_start())
            # did they return close to where they left?
            # did they return in a reasonable amount of time?
            if d < 100 and time.time() - self.recently_left.getEnd_time() < 6:

                # check to see if they've arrived at their expected destination before trying to reuse here
                # if they arrived at the predicted camera then they are probably not returning to this one
                a = self.loadActivityDb(self.recently_left.getID())
                if not a.get_has_arrived():
                    # since that is not the case, lets reuse the previous tracking record and unset the end time and predicted next camera
                    t = self.recently_left
                    t.setEnd_time(None)
                    t.setNext_camera_id(None)
                    self.saveRecoveredActivity(t)

                # blank out the recently_left field to indicate that we no longer expect someone to return soon
                self.recently_left = None

        # if no previous activity found then create a new one
        if not t:
            t = ActivityDbRow()
            t.setCamera_id(self.cameraDetails.getID())
            t.setLabel(self.get_label())
            t.setRect_start(rect_start)
            t.setStart_time(time.time())
            self.insertActivity(t)

        # keep track of the activity
        self.tracked_list.append(t)

        return t

    # this simple calculation decides if a recently leaving person went left based on the fact that their
    # bottom right x coordinate is greater than the mid point of the frame
    def went_left(self, activity):
        return (activity.getRect_end()[0] > 200)

    # this simple calculation decides if a recently leaving person went right based on the fact that their
    # top left x coordinate is less than the mid point of the frame
    def went_right(self, activity):
        return (activity.getRect_start()[0] < 200)

    # method called from flask main to toggle a flag causing the start "while" loop to exit and shut down the camera
    def stop(self):
        self.shutItDown = True

    # getter method fir the capturing boolean field
    def is_capturing(self):
        return self.capturing

    # when the browser "polls" the flask app for a frame of video, it is retrieved by calling this method
    # We use a "lock" here because jpeg might be in the middle of an update by the camera thread even
    # at the same time that the browser is trying to access it.
    def get_frame(self):
        self.lock.acquire()
        bytes = self.jpeg.tobytes()
        self.lock.release()
        return bytes
# 4/11 7:36pm JD - added functions for prediction indicator


class CameraDbRow(object):
    def __init__(self, row=None):
        self.id = None
        self.ip = None
        self.left_camera_id = None
        self.right_camera_id = None
        self.is_online = None

        self.has_motion = False
        self.has_predicted_motion = False

        if row:
            self.id = row[0]
            self.ip = row[1]
            self.left_camera_id = row[2]
            self.right_camera_id = row[3]
            self.is_online = row[4] == 'T'

    def getID(self):
        return self.id

    def setID(self, id):
        self.id = id

    def getIP(self):
        return self.ip

    def setIP(self, ip):
        self.ip = ip

    def getLeftCamera(self):
        return self.left_camera_id

    def setLeftCameraID(self, id):
        self.left_camera_id = id

    def getRightCameraID(self):
        return self.right_camera_id

    def setRightCameraID(self, id):
        self.right_camera_id = id

    def isOnline(self):
        return self.is_online

    def setIsOnline(self, online):
        self.is_online = online

    def getSelectStatement(self):
        return "select id, camera_IP, left_cam_id, right_cam_id, is_online from camera where id = %s" % self.id

    def getUpdateStatement(self):
        return "update camera set camera_IP = '%s', left_cam_id = %s, right_cam_id = %s, is_online = '%s' where id = %s" % (self.ip, (self.left_camera_id if self.left_camera_id else 'null'), (self.right_camera_id if self.right_camera_id else 'null'), ('T' if self.is_online else 'F'), self.id)

    def getInsertStatement(self):
        return "insert into camera (id, camera_IP, left_cam_id, right_cam_id) values(%s, '%s', %s, %s)" % (self.id, self.ip, (self.left_camera_id if self.left_camera_id else 'null'), (self.right_camera_id if self.right_camera_id else 'null'))

    def hasMotion(self):
        return self.has_motion

    def setHasMotion(self, b):
        self.has_motion = b

    def hasPredictedMotion(self):
        return self.has_predicted_motion

    def setHasPredictedMotion(self, b):
        self.has_predicted_motion = b
import threading
import io
import socket
import struct
import re

data_path = '/var/pispy/'
i = 0
serial = ''
socket_info = ('104.208.29.179', 27008)


def transmit(data, timestamp):
    header_size = 24  # in bytes
    args = ['iid16b', len(data), header_size, timestamp]
    args += serial
    header = struct.pack(*args)
    try:
        client = socket.socket()
        client.connect(socket_info)
        client.send(header)
        client.send(data)
        client.shutdown(socket.SHUT_RDWR)
        client.close()
    except Exception as e:
        print(e)
        print('could not connect to network')
        pass  # TODO deal with saving things if we can't talk


def read_camera(path):
    print(path)
    with io.open(path, 'rb') as f:
        while True:
            print(('reading ' + path))
            struct_format = 'id'
            header = f.read(struct.calcsize(struct_format))
            header = struct.unpack(struct_format, header)
            length = header[0]
            time = header[1]
            transmit(f.read(length), time)


def read_microphone(path):
    print(path)


def get_serial():
    # don't check for errors because this needs to work or we need to know it didn't
    f = open('/proc/cpuinfo')
    match = re.search('Serial\s*: ([0-9a-f]{16})', f.read())
    global serial
    serial = bytes(match.groups()[0], 'utf-8')


get_serial()

data_pipes = [
    (data_path + 'camera/data', read_camera),
    (data_path + 'microphone/data', read_microphone),
]

for part, function in data_pipes:
    threading.Thread(target=read, args=(f,)).start()
import sys
import cv2
import os
import time
import dlib
import imutils
from gesture import HandGestureRecognition
import numpy as np
from imutils.video import WebcamVideoStream

font = cv2.FONT_HERSHEY_SIMPLEX

#cam = WebcamVideoStream(src=0).start()
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
path_model = "hand_classifier.caffemodel"
path_proto = "hand_classifier.prototxt"
net = cv2.dnn.readNetFromCaffe(path_proto, path_model)
rec = HandGestureRecognition()
background = None
num_frames = 0


def runAverage(frame):
    #frame = cv2.medianBlur(frame,5)
    #frame = cv2.GaussianBlur(frame, (5, 5), 0)
    #frame = cv2.equalizeHist(frame)
    global num_frames, background
    if num_frames < 30:
        if background is None:
            background = frame.copy().astype("float")

        cv2.accumulateWeighted(frame, background, 0.5)
        num_frames += 1


def deepHand(image):
    d = 227
    image = cv2.resize(image, (d, d))
    blob = cv2.dnn.blobFromImage(image, 1.0, (d, d), (104.0, 177.0, 123.0))
    net.setInput(blob)
    results = net.forward()
    return results[0]


while (True):
    #frame = cam.read()
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    #h = cv2.equalizeHist(h)
    #h = cv2.medianBlur(h,3)
    #h = cv2.GaussianBlur(h, (3, 3), 100)
    runAverage(h)

    cv2.rectangle(frame, (350, 50), (600, 300), (255, 0, 0), 2)

    roi = h[50:300, 350:600]

    difference = cv2.absdiff(background.astype("uint8")[50:300, 350:600],
                             roi)
    foreground = cv2.threshold(difference, 3, 255, cv2.THRESH_BINARY)[1]
    try:
        gest, foreground = rec.recognize(foreground, frame[50:300, 350:600])
        frame[50:300, 350:600] = foreground
    except:
        pass

    frame = cv2.resize(frame, (1280, 1024))
    cv2.imshow("Cam feed", frame)
    cv2.waitKey(33)

# def bgDetect():
# return
import sys
import getopt
import math
import array
import os


raw_data = []  # list hold all numbers
list_args = []  # list holds cmd arguments
fileNames = []  # hold all file names for readInput function
results = []  # hold all unique raw_data - final result.
picData = []  # hold all number of 1 picture
numOfPic = 0
UniNum = 0
directory = ""


# get cmd args into an array
# cmd args: FaceLandmarkImg, number of image
def cmdArguments():
    for arg in sys.argv:
        list_args.append(arg)  # put command line args into a list

# read all numbers of each image
# put in the same array, convert into float type


def readInput(fileNames):
    lineNum = 0
    global numOfPic
    global pointOfPic
    numOfPic = -1
    for name in fileNames:
        with open(name) as lines:
            lineNum = 0
            raw_data.append([])
            numOfPic = numOfPic + 1
            for line in lines:
                lineNum = lineNum + 1
                if lineNum > 3 and lineNum < 72:  # just read in the numbers
                    a = line.split()  # split by "space" for each input line- read in str NOT int
                    for n in range(0, len(a)):
                        a[n] = float(a[n])
                        raw_data[numOfPic].append(a[n])

# create list of file names for readInput function
# number of file names base on the number on cmd line.
# all file names are in fileNames array


def inputFileName():
    for name in os.listdir(list_args[1]):
        fileNames.append(list_args[1] + "\\" + name)


def UNGenerator():
    global breakPoint
    global UniNum
    global numOfPoint

    for i in range(0, len(raw_data)):
        for n in range(0, len(raw_data[i])):
            raw_data[i][n] = (math.atan(raw_data[i][n]) + (math.pi / 2))
        for n in range(0, len(raw_data[i])):
            UniNum += (10 ^ n) * raw_data[i][n]
        results.append(UniNum)
        UniNum = 0

# unique number for pic
# print out the result
# results array holds all unique numbers


def UNforPic():
    for x in range(0, len(results)):
        print((fileNames[x] + "- Unique ID: " + str(results[x])))

    with open("Output.txt", "w") as text_file:
        for x in range(0, len(results)):
            text_file.write(
                fileNames[x] + "- Unique ID: " + str(results[x]) + "\n")


def testing():
    duplicate = "No duplicate"
    for x in range(0, len(results)):
        for y in range(x+1, len(results)):
            if results[x] == results[y]:
                duplicate = "Duplicate"
    print(duplicate)


# main
cmdArguments()
inputFileName()
readInput(fileNames)

# print(raw_data[2])
UNGenerator()
UNforPic()
testing()
# for m in range(0, len(fileNames)):
# print(fileNames[m])
import cv2
import numpy as np
import math


class HandGestureRecognition:
    """
    """

    def __init__(self):
        """
        """
        self.kernel = kernel = np.ones((3, 3), np.uint8)
        self.angle_cuttoff = 80.0

    def recognize(self, img, disp):
        """
        """
        segment = self._segmentHand(img)

        contours, defects = self._findHullDefects(segment)
        return self._detectGesture(contours, defects, disp)

    def _segmentHand(self, img):
        """
        """

        mask = cv2.erode(img, self.kernel, iterations=2)
        mask = cv2.dilate(mask, self.kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (3, 3), 100)
        mask = cv2.erode(mask, self.kernel, iterations=2)
        mask = cv2.dilate(mask, self.kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        cv2.imshow("Mask", mask)
        return mask

    def _findHullDefects(self, segment):
        """
        """
        _, contours, hierarchy = cv2.findContours(
            segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=lambda x: cv2.contourArea(x))
        epsilon = 0.01*cv2.arcLength(max_contour, True)
        max_contour = cv2.approxPolyDP(max_contour, epsilon, True)

        hull = cv2.convexHull(max_contour, returnPoints=False)
        defects = cv2.convexityDefects(max_contour, hull)

        return (max_contour, defects)

    def _detectGesture(self, contours, defects, img):
        """
        """
        if defects is None:
            return ['0', img]

        if len(defects) <= 2:
            return ['0', img]

        num_fingers = 1

        for i in range(defects.shape[0]):
            start_idx, end_idx, farthest_idx, _ = defects[i, 0]
            start = tuple(contours[start_idx][0])
            end = tuple(contours[end_idx][0])
            far = tuple(contours[farthest_idx][0])

            cv2.line(img, start, end, [0, 255, 0], 2)

            if angleRad(np.subtract(start, far),
                        np.subtract(end, far)) < deg2Rad(self.angle_cuttoff):
                num_fingers += 1

                # draw point as green
                cv2.circle(img, far, 5, [0, 255, 0], -1)
            else:
                # draw point as red
                cv2.circle(img, far, 5, [0, 0, 255], -1)

        return (min(5, num_fingers), img)


def angleRad(v1, v2):
    """Convert degrees to radians
    This method converts an angle in radians e[0,2*np.pi) into degrees
    e[0,360)
    """
    return np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))


def deg2Rad(angle_deg):
    """Angle in radians between two vectors
    returns the angle (in radians) between two array-like vectors
    """
    return angle_deg/180.0*np.pi
import subprocess
import time

# returns [temperature, humidity, timeCaptured]


def getHumiture():
    cmd = ['sudo', 'humiture', '11', '18']

    sthp = subprocess.check_output(cmd)
    sth = str(sthp).split(' ')
    status = sth[0]
    while status != "b'OK":
        print("Failed to read from sensor, trying again.")
        time.sleep(1)
        sthp = subprocess.check_output(cmd)
        sth = str(sthp).split(' ')
        status = sth[0]

    sth.append(str(round(time.time())))
    sth[2] = sth[2][:len(sth[2]) - 1]
    sth = sth[1:]
    return sth
f = open('a.pcm', 'rb')

array = []
array += f.read()


def cut(array, big, small):
    # caps the biggest and smallest values
    pivot = 0
    quiet = 128
    adjustment = pivot - quiet
    clipped_array = [max(min(val, big), small) for val in array]
    for i in range(len(clipped_array)):
        # print(clipped_array[i])
        # adjust for neutral value
        new_val = clipped_array[i] + adjustment
        # amplify
        new_val = new_val * 128 / (big - small)
        # back to unsigned
        clipped_array[i] = new_val + quiet
        if (clipped_array[i] > 255 or clipped_array[i] < 0):
            print((clipped_array[i]))
    return clipped_array


def low_pass_filter(array):
    p_filter = [1/6, 2/3, 1/6]
    #p_filter = [1/12, 1/6, 1/2, 1/6, 1/12]
    f_len = len(p_filter)
    assert f_len % 2 == 1

    middle = int(f_len / 2) + 1

    filtered_array = [0] * len(array)
    for i in range(middle - 1, len(array) - (middle - 1)):
        total = 0
        for j in range(middle - f_len, f_len - middle + 1):
            total += p_filter[j + middle - 1] * array[i + j]
        filtered_array[i] = round(total)
    return filtered_array


diff = 30
middle = 128
array = cut(array, middle + diff, middle - diff)
# instead of low pass perhaps we should try median pass
array = low_pass_filter(array)
f2 = open('improved.raw', 'wb')
f2.write(bytearray(array))
f2.close()
"""
Program: IRSpectrum.py
Programmed by: Josh Ellis, Josh Hollingsworth, Aaron Kruger, Alex Matthews, and
    Joseph Sneddon
Description: This program will recieve an IR Spectrograph of an unknown
    molecule and use our algorithm to compare that graph to a stored database of
    known molecules and their IR Spectrographs. This program will then return a
    list of the closest Spectrographs matches as determined by our algorithm.
IR_Functions.py: This part of the program contains most of the functions used by
    Query.py and UpdatedDB.py.
"""
# ---------------------------------Imports--------------------------------------
import PyPDF2
import sqlite3
from PIL import Image
import sys
import warnings
import os

warnings.filterwarnings("ignore")
# ------------------------------------------------------------------------------

# ---------------------------------Variables------------------------------------

# ------------------------------------------------------------------------------

# ---------------------------------Classes/Functions----------------------------


def PullImages(filename):
    '''
    Pull graph image from first page of PDF
    '''
    file = PyPDF2.PdfFileReader(open(filename, "rb"))
    xObject = file.getPage(0)

    xObject = xObject['/Resources']['/XObject'].getObject()

    images = []

    for obj in xObject:

        if xObject[obj]['/Subtype'] == '/Image':
            size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
            data = xObject[obj]._data
            if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
                mode = "RGB"
            else:
                mode = "P"

            if xObject[obj]['/Filter'] == '/FlateDecode':
                img = Image.frombytes(mode, size, data)
                img.save(filename + ".png")
                images += [filename + ".png"]
            elif xObject[obj]['/Filter'] == '/DCTDecode':
                img = open(filename + ".jpg", "wb")
                img.write(data)
                img.close()
                images += [filename + ".jpg"]
            elif xObject[obj]['/Filter'] == '/JPXDecode':
                img = open(filename + ".jp2", "wb")
                img.write(data)
                img.close()
                images += [filename + ".jp2"]
    return images


def PullStructure(filename):
    '''
    Pulls the image of the molecular structure from page 2 as a png
    '''
    file = PyPDF2.PdfFileReader(open(filename, "rb"))
    xObject = file.getPage(1)

    xObject = xObject['/Resources']['/XObject'].getObject()

    images = []

    for obj in xObject:
        if xObject[obj]['/Subtype'] == '/Image':
            size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
            data = xObject[obj].getData()
            if xObject[obj]['/Filter'] == '/FlateDecode':
                img = Image.frombytes("P", size, data)
                img.save(filename.split('.')[0] + ".png")
                images += [filename.split('.')[0] + ".png"]
    return images


def PullText(filename):
    '''
    Pull text from the first page of a PDF
    returns an array containing:
    [ SpectrumID, CAS Number, Molecular Formula, Compound Name ]
    '''
    specID = ""
    cas = ""
    formula = ""
    name = ""

    try:
        file = PyPDF2.PdfFileReader(open(filename, "rb"))
        page = file.getPage(0)

        page_content = page.extractText()

        idIndex = page_content.find("Spectrum ID")
        casIndex = page_content.find("CAS Registry Number")
        formulaIndex = page_content.find("Formula")
        nameIndex = page_content.find("CAS Index Name")
        sourceIndex = page_content.find("Source")
        startIndex = casIndex

        begin = idIndex + 11
        end = casIndex
        while begin != end:
            specID += page_content[begin]
            begin += 1

        begin = casIndex + 19
        end = formulaIndex
        while begin != end:
            cas += page_content[begin]
            begin += 1

        begin = formulaIndex + 7
        end = nameIndex
        while begin != end:
            formula += page_content[begin]
            begin += 1

        begin = nameIndex + 14
        end = sourceIndex
        while begin != end:
            name += page_content[begin]
            begin += 1
    except:
        print("There was an error extracting text from the PDF")

    # return [specID, cas, formula, name]
    return {"spectrumID": specID, "cas": cas, "formula": formula, "name": name}


def CleanStructure(filename):
    '''
    Changes all of the brightest pixels in a compound structure image
    to full alpha
    '''
    img = Image.open(filename)
    imgdata = list(img.getdata())  # the pixels from the image

    img = Image.new('RGBA', (img.size[0], img.size[1]))

    imgdata = [(i, i, i, 255) if i < 31 else (i, i, i, 0) for i in imgdata]

    img.putdata(imgdata)
    img.save(filename)


def ReadComparisonKeys():
    f = open("public\\types.keys", 'r')
    transformTypes = f.readlines()

    f.close()
    transformTypes = [line for line in
                      [lines.strip() for lines in transformTypes]
                      if len(line)]

    return transformTypes


class ReadGraph:
    '''
    Reads each datapoint in the graph and converts it to an x,y coordinate
    Each datapoint gets added to a list and returned
    '''

    def __new__(self, image):
        '''area of image scanned for data'''
        self.image = image
        self.xMin = 200
        self.xMax = 4100
        self.xRange = self.xMax-self.xMin  # the x-range of the graph.
        self.yMin = 1.02
        self.yMax = -0.05
        self.yRange = self.yMax-self.yMin  # the y-range of the graph.
        # This is the width and height standard for all IR samples
        self.width = 1024
        self.height = 768
        # the area of each image that we want (the graph)
        self.targetRect = (113, 978, 29, 724)  # (left,right,top,bottom)

        return self.readGraph(self)

    # copies pixels from the source image within the targetRect
    def cropRect(self, source):
        left, right, top, bottom = self.targetRect
        newImg = []
        for y in range(top, bottom+1):
            for x in range(left, right+1):
                newImg += [source[y*self.width+x]]
        return newImg

    # checks if the pixel at x,y is black
    def pix(self, graph, x, y):
        r, g, b = graph[y*self.width+x]
        if r+g+b >= 100:
            return False  # not black
        else:
            return True  # black

    # These two functions convert graph x,y into scientific x,y
    def convertx(self, x):
        return self.xMin+self.xRange*(x/self.width)

    def converty(self, y):
        return self.yMin+self.yRange*(y/self.height)

    def convertGraph(self, graph):
        """
        Creates a graphData list by finding each black pixel on the x axis. For each
        x get the y range over which the graph has black pixels or None if the graph
        is empty at that x value. It stores the min and max y values in the
        graphData list. Then returns the filled graphData List.
        """
        graphData = []  # to be filled with values from graph
        # For each x get the y range over which the graph has black pixels
        # or None if the graph is empty at that x value
        for x in range(0, self.width):
            graphData += [None]
            foundPix = False  # have you found a pixel while looping through the column
            for y in range(0, self.height):
                p = self.pix(self, graph, x, y)  # is the pixel black
                if p and not foundPix:
                    # record the first black pixels y value
                    foundPix = True
                    maxVal = y
                elif not p and foundPix:
                    # record the last black pixels y value
                    minVal = y
                    # write these values to data
                    graphData[-1] = (minVal, maxVal)
                    break  # next x

        return graphData

    # convert graph into datapoints
    def cleanData(self, graphData):
        data = []
        for x in range(len(graphData)):
            # Points in format x,y
            if graphData[x]:
                data += [(self.convertx(self, x),
                          self.converty(self, graphData[x][1]))]

        return data

    def readGraph(self,):
        # Crops the image
        img = Image.open(self.image)
        imgdata = list(img.getdata())  # the pixels from the image

        # The graph is cut out of the larger image
        graph = self.cropRect(self, imgdata)

        # width and height of out cropped graph
        self.width = self.targetRect[1]-self.targetRect[0]+1
        self.height = self.targetRect[3]-self.targetRect[2]+1

        # Fills graphData with values from 'graph'
        graphData = self.convertGraph(self, graph)

        # return only x,maxy and skip any none values
        data = self.cleanData(self, graphData)
        return data


def ConvertQuery(l, comparisonTypes):
    '''for each type being processed, convert the query
    and add the result to a dictionary to be returned'''
    queryDict = {}
    for cType in comparisonTypes:
        queryDict[cType] = []
        queryDict[cType] += Convert(l, cType)
    return queryDict


class Convert():
    '''
    takes the raw data and converts it into a format that can be compared later
    '''

    def __new__(self, raw, cType):
        if "raw" == cType:
            # if the comparison is by raw
            return raw
        elif '.' in cType:
            # convert the raw data into the appropiate format
            if cType.split('.')[0] == "Cumulative":
                return self.Cumulative(self, raw, int(cType.split('.')[-1]))
            elif cType.split('.')[0] == "CumulativePeak":
                return self.CumulativePeak(self, raw, int(cType.split('.')[-1]))
            elif cType.split('.')[0] == "AbsoluteROC":
                return self.AbsoluteROC(self, raw, int(cType.split('.')[-1]))
        raise ValueError("Convert type not found: "+str(cType))

    def Cumulative(self, raw, scanrange):
        '''
        The value at x=i will be total/divisor
        where total equals the sum of the points from i-scanrange to i
        and divisor equals the points from i-scanrange to i+scanrange
        '''
        raw = ['x']+raw[:]+['x']
        divisor = 0
        total = 0
        for i in range(1, scanrange+1):
            divisor += max(0.1, raw[i][1])
        retlist = []
        for i in range(1, len(raw)-1):

            low = max(0, i-scanrange)
            high = min(len(raw)-1, i+scanrange)

            total -= max(0.1, raw[low][1]) if raw[low] != "x" else 0
            total += max(0.1, raw[i][1]) if raw[i] != "x" else 0

            divisor -= max(0.1, raw[low][1]) if raw[low] != "x" else 0
            divisor += max(0.1, raw[high][1]) if raw[high] != "x" else 0

            retlist += [(raw[i][0], total/divisor)]

        return retlist

    def CumulativePeak(self, raw, scanrange):  # peak to peak transformation
        '''
        Find all peaks in list l
        Weight peaks by their height and how far they are from other taller peaks
        '''
        retlist = []
        lenl = len(raw)
        for i in range(lenl):

            # current x and y values for point i in list l
            curx = raw[i][0]
            cury = raw[i][1]

            # If this point has the same y value as the previous point
            # then continue to the next point
            if i-1 >= 0:
                if (raw[i-1][1] == cury):
                    retlist += [(curx, 0)]
                    continue

            # Search right of the point until you run into another peak or off the graph
            # sum the difference between cury and the graph at i+j to find the area right of the peak

            s1 = 0
            j = 1
            while i+j < lenl and raw[i+j][1] <= cury and j < scanrange:
                s1 += (cury - raw[i+j][1]) * (raw[i+j][0]-raw[i+j-1][0])
                j += 1

            # Same opperation but searching left
            s2 = 0
            j = -1
            while i+j >= 0 and raw[i+j][1] <= cury and j > -scanrange:
                s2 += (cury - raw[i+j][1]) * (raw[i+j+1][0]-raw[i+j][0])
                j -= 1

            # take the lowest of the 2 values
            retlist += [(curx, min(s1, s2)*cury)]

        return self.Cumulative(self, retlist, scanrange)

    def AbsoluteROC(self, raw, scanrange):
        '''
        The absolute value of the slope of the curve in list l
        Note: this method may not be useful for matching compounds
        '''
        retlist = []
        for i in range(len(raw)-1):
            retlist += [(raw[i][0], abs(raw[i+1][1]-raw[i][1]))]

        return self.Cumulative(self, retlist, scanrange)


class Compare():
    '''
    Compares a query to a subject in the database
    Converts the subject first if needed
    '''

    def __new__(self, cType, subject, query):
        if not "raw" in cType or "raw" == cType:
            # if the subject doesn't need to be converted
            return self.directCompare(self, subject, query)
        elif "." in cType:
            # else the subject need to be converted
            if cType.split('.')[0] in ["Cumulative", "CumulativePeak", "AbsoluteROC"]:
                return self.directCompare(self, Convert(subject, cType), query)
        raise ValueError("Compare type not found: "+str(cType))

    def directCompare(self, transformation1, transformation2):
        '''compares the each x in t1 to the closest x in t2'''
        difference = 0
        # Swap if needed, want t1 to be sorter than t2
        if len(transformation1) > len(transformation2):
            tmp = transformation1[:]
            transformation1 = transformation2[:]
            transformation2 = tmp

        x2 = 0
        for x1 in range(len(transformation1)):
            while transformation1[x1][0] > transformation2[x2][0] and x2 < len(transformation2)-1:
                x2 += 1
            difference += abs(transformation1[x1][1]-transformation2[x2][1])

        return difference


def AddSortResults(differenceDict, casNums):
    '''
    Take a dictionary with casNums as keys filled with dictionaries with types as keys
    Add the differences of the types for each casnum together and return a sorted list
    where each element is the compound's difference from the query followed by the casnum
    '''
    comparisonTypes = list(differenceDict.keys())[:]

    differenceList = []
    for i in range(len(casNums)):
        dif = 0
        for cType in comparisonTypes:
            if differenceDict[cType][i]:
                dif += differenceDict[cType][i][0]
        differenceList += [(dif, differenceDict[cType][i][1])]
    differenceList.sort()

    return differenceList


def SmartSortResults(differenceDict, casNums):
    '''
    Take a dictionary with casNums as keys filled with dictionaries with types as keys
    Return a sorted list where each element is the compound's difference from
    the query followed by the casnum

    The compounds are sorted by first seperating compounds by type and then sorting each list
    Each list adds its top result to the bestDict, then any compounds that have been paced
    in the bestDict by the majority of the comparison types are added to the bottom of the
    difference list
    '''
    comparisonTypes = list(differenceDict.keys())[:]

    for cType in comparisonTypes:
        differenceDict[cType].sort()
    differenceList = []

    bestDict = {}
    for i in range(len(casNums)):  # casNum
        bestDict[casNums[i]] = []

    for i in range(len(casNums)):
        tempList = []
        for cType in comparisonTypes:
            # not found due to active update
            if differenceDict[cType][i] != (0,):
                if bestDict[differenceDict[cType][i][1]] != "Done":
                    bestDict[differenceDict[cType][i][1]
                             ] += [(differenceDict[cType][i][0], cType)]
        for casNum in list(bestDict.keys()):
            if bestDict[casNum] != "Done":
                if len(bestDict[casNum]) >= max(1, len(comparisonTypes)//2+1):
                    dif = 0
                    for comp in bestDict[casNum]:
                        dif = max(dif, comp[0])
                    tempList += [(dif, casNum)]
                    bestDict[casNum] = "Done"
        if tempList:
            tempList.sort()
            differenceList += tempList

    return differenceList


class IRDB:
    def __init__(self):
        self.conn = sqlite3.connect(os.path.realpath("IR.db"))
        self.cur = self.conn.cursor()

    def searchIRDB(self, sqlQuery):
        self.cur.execute(sqlQuery)
        return self.cur.fetchall()

    def writeIRDB(self, sqlWrite, dbValues=None):
        try:
            if dbValues:
                self.cur.execute(sqlWrite, dbValues)
            else:
                self.cur.execute(sqlWrite)
            return True
        except Exception as e:
            return False

    def commitIRDB(self):
        try:
            self.conn.commit()
            return True
        except Exception as e:
            return False

    def fetchallIRDB(self):
        return self.cur.fetchall()
# ------------------------------------------------------------------------------
import socket
import time
import serialNumber
import queue
import humiture

#server = ('104.208.29.115', 27007)
server = ('104.208.39.124', 27007)

clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientsocket.connect(server)
clientsocket.settimeout(5)

serial = serialNumber.getSerialNumber()
q = queue.Queue()

humitureCmd = ['sudo', 'humiture', '11', '18']

while True:
    q.put(humiture.getHumiture())  # [temperature, humidity, timeCaptured]
    while not q.empty():

        msg = q.get()
        message = (serial + ";" + msg[0] + ";" + msg[1] + ";" + msg[2])
        print(("Attempting to send to server: " + message))

        try:
            numSent = clientsocket.send(bytes(message, "UTF-8"))
            print(("Sent " + str(numSent) + " of " + str(len(message)) + " bytes."))
            if str(numSent) != str(len(message)):
                print("Message not fully sent, reqeueing message now.")
                q.put(msg)
                break

        except Exception as e:
            print(("Failed to send message. Error: " + str(e)
                   + "\nAttempting to reestablish connection."))
            q.put(msg)
            try:
                clientsocket.close()
                print("Socket Released.")
            except Exception as e:
                print(("Failed to release socket: " + str(e)))
            finally:
                try:
                    clientsocket = socket.socket(
                        socket.AF_INET, socket.SOCK_STREAM)
                    clientsocket.connect(server)
                    print("Connection reestablished.")
                except Exception as e2:
                    print(("Failed to reestablish connection: " + str(e2)))
            break

    time.sleep(60)

clientsocket.close()
import socket
import time
import serialNumber
import queue
import random

#server = ('104.208.29.115', 27007)
server = ('104.208.39.124', 27007)

clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientsocket.connect(server)
clientsocket.settimeout(5)

serial = serialNumber.getSerialNumber()
q = queue.Queue()

while True:
    q.put([str(random.randrange(18.0, 25.0)), str(random.randrange(0.0, 100.0)), str(
        round(time.time()))])  # [temperature, humidity, timeCaptured]
    while not q.empty():

        msg = q.get()
        message = (serial + ";" + msg[0] + ";" + msg[1] + ";" + msg[2])
        print(("Attempting to send to server: " + message))

        try:
            numSent = clientsocket.send(bytes(message, "UTF-8"))
            print(("Sent " + str(numSent) + " of " + str(len(message)) + " bytes."))
            if str(numSent) != str(len(message)):
                print("Message not fully sent, reqeueing message now.")
                q.put(msg)
                break

        except Exception as e:
            print(("Failed to send message. Error: " + str(e)))
            q.put(msg)
            try:
                clientsocket.close()
                print("Socket Released.")
            except Exception as e:
                print(("Failed to release socket: " + str(e)))
            finally:
                try:
                    clientsocket = socket.socket(
                        socket.AF_INET, socket.SOCK_STREAM)
                    clientsocket.connect(server)
                    print("Connection reestablished.")
                except Exception as e2:
                    print(("Failed to reestablish connection: " + str(e2)))
            break

    time.sleep(60)

clientsocket.close()
# import the necessary packages
from pyimagesearch.tempimage import TempImage
from dropbox.client import DropboxOAuth2FlowNoRedirect
from dropbox.client import DropboxClient
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import warnings
import datetime
import imutils
import json
import time
import cv2
import time

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
                help="path to the JSON configuration file")
args = vars(ap.parse_args())

# filter warnings, load the configuration and initialize the Dropbox
# client
warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))
client = None

if conf["use_dropbox"]:
    # connect to dropbox and start the session authorization process
    flow = DropboxOAuth2FlowNoRedirect(
        conf["dropbox_key"], conf["dropbox_secret"])
    print(("[INFO] Authorize this application: {}".format(flow.start())))
    authCode = input("Enter auth code here: ").strip()

    # finish the authorization and grab the Dropbox client
    (accessToken, userID) = flow.finish(authCode)
    client = DropboxClient(accessToken)
    print("[SUCCESS] dropbox account linked")


# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = tuple(conf["resolution"])
camera.framerate = conf["fps"]
rawCapture = PiRGBArray(camera, size=tuple(conf["resolution"]))

# allow the camera to warmup, then initialize the average frame, last
# uploaded timestamp, and frame motion counter
print("[INFO] warming up...")
time.sleep(conf["camera_warmup_time"])
avg = None
lastUploaded = datetime.datetime.now()
motionCounter = 0

# capture frames from the camera
for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image and initialize
    # the timestamp and occupied/unoccupied text
    frame = f.array
    timestamp = datetime.datetime.now()
    text = "Unoccupied"

    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # if the average frame is None, initialize it
    if avg is None:
        print("[INFO] starting background model...")
        avg = gray.copy().astype("float")
        rawCapture.truncate(0)
        continue

    # accumulate the weighted average between the current frame and
    # previous frames, then compute the difference between the current
    # frame and running average
    cv2.accumulateWeighted(gray, avg, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    # threshold the delta image, dilate the thresholded image to fill
    # in holes, then find contours on thresholded image
    thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255,
                           cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < conf["min_area"]:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied"

    # draw the text and timestamp on the frame
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")

    cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (0, 0, 255), 1)

    # draw temperature on the screen
    infile = open("tempOutput.txt", "r")
    temp = infile.readline().rstrip()
    #print('received temp of: ' + temp)

    cv2.putText(frame, "Room Temperature: {}".format(temp), (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, temp, (50, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (0, 0, 255), 1)

    # check to see if the room is occupied
    if text == "Occupied":
        #targetFile = open('videoOutputFile', 'w')
        # targetFile.write('yes')
        # NEED TO CALL AZURE CLOUD HERE INSTEAD OF PRINTING TO FILE
        # SEND YES TO SURVICE BUS

        # check to see if enough time has passed between uploads
        if (timestamp - lastUploaded).seconds >= conf["min_upload_seconds"]:
            # increment the motion counter
            motionCounter += 1

            # check to see if the number of frames with consistent motion is
            # high enough
            if motionCounter >= conf["min_motion_frames"]:
                # check to see if dropbox sohuld be used
                if conf["use_dropbox"]:
                    # write the image to temporary file
                    t = TempImage()
                    cv2.imwrite(t.path, frame)

                    # upload the image to Dropbox and cleanup the tempory image
                    print(("[UPLOAD] {}".format(ts)))
                    path = "{base_path}/{timestamp}.jpg".format(
                        base_path=conf["dropbox_base_path"], timestamp=ts)
                    client.put_file(path, open(t.path, "rb"))
                    t.cleanup()

                # update the last uploaded timestamp and reset the motion
                # counter
                lastUploaded = timestamp
                motionCounter = 0

    # otherwise, the room is not occupied
    else:
        targetFile = open('videoOutputFile', 'w')

    # check to see if the frames should be displayed to screen
    if conf["show_video"]:
        # display the security feed
        cv2.imshow("Security Feed", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
"""
Program: IRSpectrum.py
Programmed by: Josh Ellis, Josh Hollingsworth, Aaron Kruger, Alex Matthews, and
    Joseph Sneddon
Description: This program will recieve an IR Spectrograph of an unknown
    molecule and use our algorithm to compare that graph to a stored database of
    known molecules and their IR Spectrographs. This program will then return a
    list of the closest Spectrographs matches as determined by our algorithm.
Query.py: This part of the program recieves the file location of a query IR
    Spectrograph downloaded by main.js. formatQueryData() then formats the query
    data and returns a dictionary, queryDict, of the formated query data.
    compareQueryToDB() then takes that dictionary and compares it against all of
    the IR spectrographs imported from our IR spectrum database (IR.db).
    compareQueryToDB() then sends a string back to main.js of the closest IR
    spectrographs found in the IR.db.
"""
# ---------------------------------Imports--------------------------------------
import sys
import os
from IR_Functions import *
import multiprocessing as mp
from shutil import copyfile
# ------------------------------------------------------------------------------

# ---------------------------------Classes/Functions----------------------------


class FormatQueryData:
    def __new__(self, queryPath, comparisonTypes, filename):
        """
        Creates class object and initializes class variables, then returns a
        dictionary of the formated query data.
        """
        self.queryPath = queryPath
        self.comparisonTypes = comparisonTypes
        self.filename = filename

        return self.formatQueryData(self)

    def timeStamp(self, f):
        return int(f.split('.')[0].split('_')[-1])

    def cleanupQueryData(self, images):
        """Removes all generated query data that is more than 5 min old."""
        currentTime = self.timeStamp(self, self.filename)
        holdTime = 2*60*1000
        for each in [file for file in os.listdir("public\\uploads")
                     if file.endswith(".jpg")]:
            try:
                if self.timeStamp(self, each) < currentTime-holdTime:
                    os.remove("public\\uploads\\"+each)
            except:
                pass

        # Deletes the temp file downloaded by main.js
        os.remove(images[0])
        if 'temp' in self.queryPath:
            os.remove(self.queryPath)

    def formatQueryData(self):
        # Open the source image
        # PullImages() from IR_Functions.py
        images = PullImages(self.queryPath)
        IR_Data = ReadGraph(images[0])  # ReadGraph() from IR_Functions.py

        copyfile(images[0], "public\\uploads\\" + self.filename)

        # Cleans up temp data from queries.
        self.cleanupQueryData(self, images)

        # Calculate each transformation. ConvertQuery() from IR_Functions.py
        queryDict = ConvertQuery(IR_Data, self.comparisonTypes)

        return queryDict
# ------------------------------------------------------------------------------

# ----------------------------Multiprocessing functions-------------------------


def work(DataQ, ReturnQ, query, comparisonTypes):
    try:
        casNum, dataDict = DataQ.get()

        differenceDict = {}
        for cType in comparisonTypes:
            differenceDict[cType] = []

            dif = Compare(cType, dataDict[cType], query[cType])

            differenceDict[cType] += [(dif, casNum)]
        ReturnQ.put(differenceDict)
        return True
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        if int(exc_tb.tb_lineno) == 83:
            # error due to active update
            ReturnQ.put(None)
            return False
        print('\nERROR!:')
        print(('%s' % e))
        print(("\n"+str(exc_tb.tb_lineno)+" "+str(exc_obj)+" "+str(exc_tb), "\n"))
        return False


def worker(workerNo, JobsDoneQ, NofJobs, NofWorkers, ReturnQ, DataQ, query,
           comparisonTypes):
    # Worker loop
    working = True
    while working:
        jobNo = JobsDoneQ.get()
        work(DataQ, ReturnQ, query, comparisonTypes)
        if NofJobs-jobNo <= NofWorkers-1:
            working = False


def multiProcessController(formatedQueryData, comparisonTypes, IR_Info, dataDict, differenceDict):
    CORES = min(mp.cpu_count(), len(IR_Info))

    JobsDoneQ = mp.Queue()
    ReturnQ = mp.Queue()
    ReadRequestQ = mp.Queue()
    DataQ = mp.Queue()
    DataBuffer = min(CORES*2, len(IR_Info))

    for iCompound in range(len(IR_Info)):
        JobsDoneQ.put(iCompound+1)
        ReadRequestQ.put(1)
    for iCompound in range(DataBuffer):
        DataQ.put((IR_Info[iCompound][0], dataDict[IR_Info[iCompound][0]]))
        ReadRequestQ.get()
        ReadRequestQ.put(0)

    p = {}
    for core in range(CORES):
        p[core] = mp.Process(target=worker,
                             args=[core, JobsDoneQ, len(IR_Info), CORES, ReturnQ, DataQ,
                                   formatedQueryData, comparisonTypes])
        p[core].start()

    # Read returned data from workers, add new read reqests
    for iCompound in range(DataBuffer, len(IR_Info)+DataBuffer):
        retDict = ReturnQ.get()
        if retDict:
            for cType in comparisonTypes:
                differenceDict[cType] += retDict[cType]
        else:  # not found due to active update
            for cType in comparisonTypes:
                differenceDict[cType] += [(0,)]
        if ReadRequestQ.get():
            DataQ.put((IR_Info[iCompound][0], dataDict[IR_Info[iCompound][0]]))

    for core in range(CORES):
        p[core].join()
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------


def importDB():
    try:
        myIRDB = IRDB()
        IR_Info = myIRDB.searchIRDB(
            "SELECT CAS_Num FROM IR_Info GROUP BY CAS_Num")
        IR_Data = myIRDB.searchIRDB(
            "SELECT CAS_Num,Type,Wavelength,Value FROM IR_Data")

        return IR_Info, IR_Data
    except:
        return None


def generateDataDict(IR_Info, IR_Data, comparisonTypes):
    dataDict = {}
    for iCompound in range(len(IR_Info)):
        dataDict[IR_Info[iCompound][0]] = {}
        for cType in comparisonTypes:
            dataDict[IR_Info[iCompound][0]][cType] = []
    for iDBrow in range(len(IR_Data)):
        if 'raw' != IR_Data[iDBrow][1]:
            dataDict[IR_Data[iDBrow][0]][IR_Data[iDBrow]
                                         [1]] += [IR_Data[iDBrow][2:]]
        else:
            for cType in comparisonTypes:
                if 'raw' in cType:
                    dataDict[IR_Data[iDBrow][0]
                             ][cType] += [IR_Data[iDBrow][2:]]
    return dataDict


def generateDifDict(comparisonTypes):
    differenceDict = {}
    for cType in comparisonTypes:
        differenceDict[cType] = []
    return differenceDict


def compareQueryToDB(formatedQueryData, comparisonTypes):
    IR_Info, IR_Data = importDB()

    dataDict = generateDataDict(IR_Info, IR_Data, comparisonTypes)

    differenceDict = generateDifDict(comparisonTypes)

    multiProcessController(formatedQueryData, comparisonTypes,
                           IR_Info, dataDict, differenceDict)

    # Sort compounds by difference. SmartSortResults() from IR_Functions.py
    results = SmartSortResults(differenceDict, [a[0] for a in IR_Info])[
        :min(20, len(IR_Info))]
    retString = ""

    # Save list of compound differences to file
    for iResult in range(len(results)):
        retString += results[iResult][1]+" "

    # Gives sorted list of Output to main.js
    return retString.strip()
# ------------------------------------------------------------------------------

# ---------------------------------Program Main---------------------------------


def main(queryPath, filename):

    if importDB():
        # get comparison types from file
        comparisonTypes = ReadComparisonKeys()

        formatedQueryData = FormatQueryData(
            queryPath, comparisonTypes, filename)

        results = compareQueryToDB(formatedQueryData, comparisonTypes)
        print(results)

        sys.stdout.flush()
    else:
        print("DB_Not_Found")

        sys.stdout.flush()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
# ---------------------------------End of Program-------------------------------
# -*- coding: utf-8 -*-

import RPi.GPIO as GPIO
import time


def bin2dec(string_num):
    return str(int(string_num, 2))


data = []

GPIO.setmode(GPIO.BCM)

GPIO.setup(4, GPIO.OUT)
GPIO.output(4, GPIO.HIGH)
time.sleep(0.025)
GPIO.output(4, GPIO.LOW)
time.sleep(0.02)

GPIO.setup(4, GPIO.IN, pull_up_down=GPIO.PUD_UP)

for i in range(0, 500):
    data.append(GPIO.input(4))


bit_count = 0
tmp = 0
count = 0
HumidityBit = ""
TemperatureBit = ""
crc = ""


try:
    while data[count] == 1:
        tmp = 1
        count = count + 1

    for i in range(0, 50):
        bit_count = 0

        while data[count] == 0:
            tmp = 1
            count = count + 1

        while data[count] == 1:
            bit_count = bit_count + 1
            count = count + 1

        if bit_count > 3:
            if i >= 0 and i < 8:
                HumidityBit = HumidityBit + "1"
            if i >= 16 and i < 24:
                TemperatureBit = TemperatureBit + "1"
        else:
            if i >= 0 and i < 8:
                HumidityBit = HumidityBit + "0"
            if i >= 16 and i < 24:
                TemperatureBit = TemperatureBit + "0"

except:
    print("ERR_RANGE")
    exit(0)


try:
    for i in range(0, 8):
        bit_count = 0

        while data[count] == 0:
            tmp = 1
            count = count + 1

        while data[count] == 1:
            bit_count = bit_count + 1
            count = count + 1

        if bit_count > 3:
            crc = crc + "1"
        else:
            crc = crc + "0"
except:
    print("ERR_RANGE")
    exit(0)


Humidity = bin2dec(HumidityBit)
Temperature = bin2dec(TemperatureBit)

if int(Humidity) + int(Temperature) - int(bin2dec(crc)) == 0:
    print(Humidity)
    print(Temperature)
else:
    print("ERR_CRC")
import time
import sys
from azure.servicebus import ServiceBusService

infile = open("tempOutput.txt", "r")
temp = infile.readline().rstrip()
#print('received temp of: ' + temp)
temp = int(temp)

key_name = "sendRule"
key_value = "9SWS0sNEBQMfTmuBHlxFwUHBFMSBgmJ77/ICSRm9HK4="

sbs = ServiceBusService(
    "pimessage-ns", shared_access_key_name=key_name, shared_access_key_value=key_value)
if temp > 65 or temp < 30:
    #    print('sending temp of:' + temp)
    sbs.send_event(
        'pimessage', '{ "DeviceId": "smokerpi", "Temperature": temp }')
    print('sent!')
    print('got here')
else:
    print('temp was in normal range')
import time
import sys
from azure.servicebus import ServiceBusService

from twilio.rest import TwilioRestClient

#key_name = "sendRule"
#key_value = "9SWS0sNEBQMfTmuBHlxFwUHBFMSBgmJ77/ICSRm9HK4="

#sbs = ServiceBusService("pimessage-ns",shared_access_key_name=key_name, shared_access_key_value=key_value)

# while(True):
#	print('sending...')
#	sbs.send_event('pimessage', '{ "DeviceId": "smokerpi", "Temperature": "37.0" }')
#	print('sent!')
#	time.sleep(10)

infile = open("tempOutput.txt", "r")
temp = infile.readline().rstrip()
#print('received temp of: ' + temp)
temp = int(temp)
client = TwilioRestClient(account='AC5e63bbdefc2e7374af34ce71e9d252d7',
                          token='76910c3c9f315d9e39c914581101b969')

client.messages.create(to='+14175408907', from_='19182382589', body=temp)
time.sleep(1)
# returns the raspberry pi's serial number
def getSerialNumber():
    serial = "0000000000000000"

    f = open('/proc/cpuinfo', 'r')
    for line in f:
        if line[0:6] == 'Serial':
            serial = line[10:26]
    f.close()

    return serial
import picamera
import io
import struct
import time

resolution = (640, 480)
framerate = 10
next_header_index = 0
output = None


def write_video(stream, start_time):
    global next_header_index
    with stream.lock:
        first_frame = None
        second_frame = None
        third_frame = None
        for f in stream.frames:
            if f.header and f.index >= next_header_index:
                if second_frame and first_frame:
                    third_frame = f
                    break
                if not second_frame and first_frame:
                    second_frame = f
                if not first_frame:
                    first_frame = f
        if first_frame and second_frame and third_frame:
            print('writing')
            stream.seek(first_frame.position)
            # figure out the exact size needed
            # contents = stream.read(second_frame.position - first_frame.position)
            contents = stream.read()
            content_length = len(contents)
            header = struct.pack('id', content_length,
                                 start_time + first_frame.index / framerate)
            contents = header + contents
            output.write(contents)
            output.flush()
            next_header_index = second_frame.index


with picamera.PiCamera() as camera:
    stream = picamera.PiCameraCircularIO(camera, seconds=20)
    camera.resolution = resolution
    camera.framerate = framerate
    camera.start_recording(stream, format='h264')
    start_time = round(time.time(), 1)
    output = io.open('/var/pispy/camera/data', 'wb')
    try:
        while True:
            camera.wait_recording(1)
            write_video(stream, start_time)
    finally:
        camera.stop_recording()

# import the necessary packages
import uuid
import os


class TempImage:
    def __init__(self, basePath="./", ext=".jpg"):
        # construct the file path
        self.path = "{base_path}/{rand}{ext}".format(base_path=basePath,
                                                     rand=str(uuid.uuid4()), ext=ext)

    def cleanup(self):
        # remove the file
        os.remove(self.path)
"""
Program: IRSpectrum.py
Programmed by: Josh Ellis, Josh Hollingsworth, Aaron Kruger, Alex Matthews, and
    Joseph Sneddon
Description: This program will recieve an IR Spectrograph of an unknown
    molecule and use our algorithm to compare that graph to a stored database of
    known molecules and their IR Spectrographs. This program will then return a
    list of the closest Spectrographs matches as determined by our algorithm.
UpdateDB.py: This part of the program imports all pdf files from */IR_samples
    and updates the database (IR.db) with each new compound found.
"""
# ---------------------------------Imports--------------------------------------
import sys
import sqlite3
import os
from PIL import Image
from shutil import copyfile
import multiprocessing as mp
import time
from IR_Functions import *
# ------------------------------------------------------------------------------

# ---------------------------------Classes/Functions----------------------------


def initializeDB():
    # If IR.db somehow gets deleted then re-create it.
    if not os.path.exists("IR.db"):
        file = open('IR.db', 'w+')
        file.close()

    sqlData = "CREATE TABLE IF NOT EXISTS `IR_Data` ( `CAS_Num` TEXT, `Type` \
                TEXT, `Wavelength` NUMERIC, `Value` NUMERIC )"
    sqlInfo = "CREATE TABLE IF NOT EXISTS `IR_Info` ( `Spectrum_ID` TEXT, \
                `CAS_Num` TEXT, `Formula` TEXT, `Compound_Name` TEXT, \
                PRIMARY KEY(`Spectrum_ID`) )"

    myIRDB = IRDB()
    myIRDB.writeIRDB(sqlData)
    myIRDB.writeIRDB(sqlInfo)
    myIRDB.commitIRDB()


def tryWork(Jobs, comparisonTypes):
    try:
        file = Jobs.get()

        """ Open the source image """
        images = PullImages(file)

        fname = file.split("\\")[-1]
        casNum = fname.split(".")[0]

        """ is this file already in the database? """
        myIRDB = IRDB()
        sqlQ = "SELECT CAS_Num FROM IR_Info WHERE CAS_Num='"+casNum+"'"
        sqlData = "SELECT CAS_Num FROM IR_Data WHERE CAS_Num='"+casNum+"'"
        sqlInfo = "INSERT INTO IR_Info(Spectrum_ID, CAS_Num, Formula, \
                                        Compound_Name) VALUES (?, ?, ?, ?)"

        myIRDB.writeIRDB(sqlQ)
        myIRDB.writeIRDB(sqlData)
        qData = myIRDB.fetchallIRDB()

        """ if not in the database set the flag to add it """
        if len(qData) == 0:

            copyfile(images[0], "public\\images\\"+casNum+".jpg")

            structure = PullStructure(file)[0]
            CleanStructure(structure)
            copyfile(structure, "public\\info\\"+structure.split("\\")[-1])
            os.remove(structure)

            values = PullText(file)
            # Save compound data into the database
            dbvalues = (list(values.values())[0], casNum,
                        list(values.values())[2], list(values.values())[3])

            myIRDB.writeIRDB(sqlInfo, dbvalues)
            myIRDB.commitIRDB()

            f = open("public\\info\\"+casNum+".json", 'w')
            f.write(str(values).replace("'", '"'))
            f.close()
        else:
            os.remove(images[0])
            return casNum+" already in DB"

        data = ReadGraph(images[0])  # ReadGraph() from IR_Functions.py
        os.remove(images[0])

        # calculate each transformation
        comparisonDict = {}
        for cType in comparisonTypes:
            comparisonDict[cType] = Convert(data, cType)

        sqlQ = "INSERT INTO IR_Data(CAS_Num, Type, Wavelength, Value) \
                    VALUES (?, ?, ?, ?)"
        # save each transformation to file
        for cType in comparisonDict:
            d = []
            for row in comparisonDict[cType]:
                d += [str(row[0])+','+str(row[1])]
                dbvalues = (casNum, cType, row[0], row[1])
                myIRDB.writeIRDB(sqlQ, dbvalues)
                # save data

        myIRDB.commitIRDB()
        return casNum+" added to DB"

    except Exception as e:
        print('\nERROR!:')
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(('%s' % e))
        print(("\n"+str(exc_tb.tb_lineno)+" "+str(exc_obj)+" "+str(exc_tb), "\n"))
        return False
# ------------------------------------------------------------------------------

# ----------------------------Multiprocessing functions-------------------------


def worker(Jobs, workerNo, NofWorkers, JobsDoneQ, NofJobs, comparisonTypes):
    working = True
    while working:
        message = tryWork(Jobs, comparisonTypes)
        if message:
            jobNo = JobsDoneQ.get()
            print(("[Worker No. "+str(workerNo)+"] "+str(jobNo)+" of "
                   + str(NofJobs)+" "+message))
            if NofJobs-jobNo <= NofWorkers-1:
                working = False
        else:
            working = False


def multiProcessUpdater(comparisonTypes):
    filedir = [os.path.join("IR_samples", file) for file in
               os.listdir("IR_samples") if file.endswith(".pdf")]

    Jobs = mp.Queue()
    JobsDoneQ = mp.Queue()
    for i in range(len(filedir)):
        Jobs.put(filedir[i])
        JobsDoneQ.put(i+1)

    CORES = min(mp.cpu_count(), len(filedir))
    p = {}
    print("Starting")
    start = time.time()
    for core in range(CORES):
        p[core] = mp.Process(target=worker, args=[Jobs, core, CORES, JobsDoneQ, len(filedir),
                                                  comparisonTypes])
        p[core].start()
    for core in range(CORES):
        p[core].join()
    print(("Done and Done "+str(time.time()-start)))
# ------------------------------------------------------------------------------

# ---------------------------------Program Main---------------------------------


def main():

    comparisonTypes = ReadComparisonKeys()

    # Edits comparisonTypes to include only a single raw
    # comparisons with the raw argument will be calculated in the future.
    raws = []
    for icomp in range(len(comparisonTypes)-1, -1, -1):
        if 'raw' in comparisonTypes[icomp]:
            raws += [comparisonTypes.pop(icomp)]
    if len(raws) > 0:
        comparisonTypes += ['raw']

    initializeDB()

    multiProcessUpdater(comparisonTypes)


if __name__ == "__main__":
    main()
# ---------------------------------End of Program-------------------------------
