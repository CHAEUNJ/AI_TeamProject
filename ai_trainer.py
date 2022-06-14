import cv2

# 0, 1: 팔꿈치
# 20 ~ 60
# 2, 3: 겨드랑이
# 20 ~ 60
# 4, 5: 무릎
# 170 ~ 180, 40 ~ 180
# 6, 7: 발목
# -
# 8, 9: 고관절
# 150 ~ 175, 55 ~ 175

class AITrainer:
    def check_point(self, x, y, score):
        if score:
            cv2.circle(self.img, (x, y), 10, (51, 153, 51), cv2.FILLED)
            cv2.circle(self.img, (x, y), 15, (51, 153, 51), 2)
        else:
            cv2.circle(self.img, (x, y), 10, (51, 0, 255), cv2.FILLED)
            cv2.circle(self.img, (x, y), 15, (51, 0, 255), 2)

    def elbow_check(self):
        # 팔꿈치 각도 교정
        if 20 < self.angleList[0] < 60 and 20 < self.angleList[1] < 60 \
                and abs(self.angleList[0] - self.angleList[1]) < 10:

            #print("READY) - elbow ready!")
            self.check_point(self.lmList[3][1], self.lmList[3][2], True)
            self.check_point(self.lmList[2][1], self.lmList[2][2], True)
            return True
        else:
            self.check_point(self.lmList[3][1], self.lmList[3][2], False)
            self.check_point(self.lmList[2][1], self.lmList[2][2], False)
            return False

    def shoulder_check(self):
        # 어깨 각도 교정
        if 20 < self.angleList[2] < 60 and 20 < self.angleList[3] < 60 \
                and abs(self.angleList[2] - self.angleList[3]) < 10:

            #print("READY) - shoulder ready!")
            self.check_point(self.lmList[0][1], self.lmList[0][2], True)
            self.check_point(self.lmList[1][1], self.lmList[1][2], True)
            return True
        else:
            self.check_point(self.lmList[0][1], self.lmList[0][2], False)
            self.check_point(self.lmList[1][1], self.lmList[1][2], False)
            return False

    def knee_check(self, status):
        # 무릎 각도 교정 - 준비 시
        if status == "Ready":
            angle_1 = 170
            angle_2 = 180
        else:
            angle_1 = 40
            angle_2 = 180

        if angle_1 < self.angleList[4] < angle_2 and angle_1 < self.angleList[5] < angle_2 \
                and abs(self.angleList[4] - self.angleList[5]) < 10:

            #print("READY) - knee ready!")
            self.check_point(self.lmList[14][1], self.lmList[14][2], True)
            self.check_point(self.lmList[15][1], self.lmList[15][2], True)
            return True
        else:
            self.check_point(self.lmList[14][1], self.lmList[14][2], False)
            self.check_point(self.lmList[15][1], self.lmList[15][2], False)
            return False

    def hip_check(self, status):
        # 고관절 각도 교정 - 준비 시
        if status == "Ready":
            angle_1 = 150
            angle_2 = 175
        else:
            angle_1 = 55
            angle_2 = 175

        if angle_1 < self.angleList[8] < angle_2 and angle_1 < self.angleList[9] < angle_2 \
                and abs(self.angleList[8] - self.angleList[9]) < 10:

            #print("READY) - hip ready!")
            self.check_point(self.lmList[12][1], self.lmList[12][2], True)
            self.check_point(self.lmList[13][1], self.lmList[13][2], True)
            return True
        else:
            self.check_point(self.lmList[12][1], self.lmList[12][2], False)
            self.check_point(self.lmList[13][1], self.lmList[13][2], False)
            return False

    def check_pose(self, img, lmList, angleList, status):
        self.img = img
        self.lmList = lmList
        self.angleList = angleList

        check_list = []

        check_list.append(self.elbow_check())
        check_list.append(self.shoulder_check())
        check_list.append(self.knee_check(status))
        check_list.append(self.hip_check(status))

        return all(check_list)

