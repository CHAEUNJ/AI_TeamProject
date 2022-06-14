import cv2
import math
import mediapipe

class PoseDetector:

    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=True, trackCon=True):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mediapipe.solutions.drawing_utils

        # mediapipe 모듈의 mediapipe.solutions.pose 모듈은 pose detection을 위한 모듈임
        self.mpPose = mediapipe.solutions.pose

        # 위 설정들을 바탕으로 pose detection의 옵션을 지정함
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,
                                     self.detectionCon, self.trackCon)

    def findPose(self, img, draw=False):
        #print("-----------------findPose-----------------")
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 모델에서 pose estimation을 실행하고 이에 대한 결과값을 저장함
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            #print(self.results.pose_landmarks)
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, draw=False):
        #print("-----------------findPosition-----------------")

        self.lmList = []

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                if id < 11:
                    continue

                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        #print("-----------------findAngle-----------------")

        # Get the landmarks
        x1, y1 = self.lmList[p1-11][1:]
        x2, y2 = self.lmList[p2-11][1:]
        x3, y3 = self.lmList[p3-11][1:]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

        if angle < 0:
            angle += 360

        if angle > 180:
            angle = 360 - angle

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (153, 255, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (153, 255, 255), 2)
            cv2.circle(img, (x2, y2), 10, (153, 255, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (153, 255, 255), 2)
            cv2.circle(img, (x3, y3), 10, (153, 255, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (153, 255, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return int(angle)

    def findAcc(self, p1_before, p2_before):
        x1_before, y1_before = p1_before[1:]
        x2_before, y2_before = p2_before[1:]

        x1_after, y1_after = self.lmList[13][1:]
        x2_after, y2_after = self.lmList[12][1:]

        x_before = (x1_before + x2_before) / 2
        y_before = (y1_before + y2_before) / 2

        x_after = (x1_after + x2_after) / 2
        y_after = (y1_after + y2_after) / 2

        a = x_before - x_after
        b = y_before - y_after
        c = math.sqrt((a * a) + (b * b))

        return int(c)
