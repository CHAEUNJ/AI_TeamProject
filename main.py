import cv2
import time

from pose_detector import PoseDetector
from simple_facerec import SimpleFacerec
from ai_trainer import AITrainer

import ssl

def main():
    ssl._create_default_https_context = ssl._create_unverified_context

    count_face = 0
    fps_control = 0
    face_location_count = 0

    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture('squat_3.mp4')

    pTime = 0

    # face recognition 진행 후 아래 진행

    sfr = SimpleFacerec()
    sfr.load_encoding_images("face_images/")

    detector = PoseDetector()
    name_data = []
    user_name = [""]

    ai_trainer = AITrainer()

    print("Initialize Complete!")

    while True:
        success, img = cap.read()

        if not success:
            break

        #if fps_control % 3 == 0:
        face_locations, face_names = sfr.detect_known_faces(img, False)

        #print("face_location_count:", face_location_count)

        if len(face_locations):
            print("face_count: ", face_location_count)
            face_location_count += 1
            if face_location_count > 30:
                face_locations, face_names = sfr.detect_known_faces(img, True)
                name_data.append(face_names)

                if face_location_count > 31:
                    user_name = max(name_data, key=name_data.count)
                    break

        else:
            face_location_count = 0
            name_data = []

        #print(face_locations, face_names)

        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
            cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 200), 4)

        #fps_control += 1

        #img = cv2.resize(img, dsize=(0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
        cv2.imshow("Image", img) #cv2.flip(img, 1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Face Recognition Complete!")

    acc_count = -1
    acc_before = []
    acc_status = "Ok"

    squat_status = "Ready"
    squat_updown = "-"
    squat_count = 0

    while True:
        # 동영상에서 읽은 하나의 프레임과 성공 여부 return
        success, img = cap.read()

        if not success:
            break

        if fps_control % 2 == 0:
            # 동영상 프레임을 기반으로 총 33개의 landmark와 x, y비 및 신뢰성 계산
            img = detector.findPose(img)

            # findPose에서 구한 값을 기반으로 각 landmark에 대한 실제 x, y 좌표를 계산
            lmList = detector.findPosition(img, draw=True)
            # 원하는 특정 부위에 표시 가능 (현재 코드의 경우, 오른쪽 팔꿈치에 빨간색 원 표시)
            #if len(lmList) != 0:
            #    print(lmList[14])
            #    cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)

            # 이전 처리 시간과 현재 시간으로 현재 fps를 구해서 표시함
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            angleList = []

            if lmList:
                angleList.append(detector.findAngle(img, 16, 14, 12))   # 오른쪽 팔꿈치 각도
                angleList.append(detector.findAngle(img, 15, 13, 11))   # 왼쪽 팔꿈치 각도

                angleList.append(detector.findAngle(img, 14, 12, 24))   # 오른쪽 겨드랑이 각도
                angleList.append(detector.findAngle(img, 13, 11, 23))   # 왼쪽 겨드랑이 각도

                angleList.append(detector.findAngle(img, 23, 25, 27))   # 오른쪽 무릎 각도
                angleList.append(detector.findAngle(img, 24, 26, 28))  # 왼쪽 무릎 각도

                angleList.append(detector.findAngle(img, 25, 27, 31))  # 오른쪽 발목 각도
                angleList.append(detector.findAngle(img, 26, 28, 32))  # 왼쪽 발목 각도

                angleList.append(detector.findAngle(img, 12, 24, 26))  # 오른쪽 고관절 각도
                angleList.append(detector.findAngle(img, 11, 23, 25))  # 왼쪽 고관절 각도

                acc_count += 1

                if acc_count == 0:
                    #print(lmList)
                    #print("acc_before append!")
                    acc_before.append(lmList[13])
                    acc_before.append(lmList[12])
                elif acc_count == 2:
                    #print("acc:", detector.findAcc(acc_before[0], acc_before[1]))
                    if detector.findAcc(acc_before[0], acc_before[1]) > 40:
                        acc_status = "Fast"
                    else:
                        acc_status = "Ok"

                    acc_before = []
                    acc_count = -1

            if len(angleList) == 10:
                if squat_status == "Ready":
                    if ai_trainer.check_pose(img, lmList, angleList, squat_status):
                        squat_status = "START"
                        squat_updown = "UP"
                else:
                    ai_trainer.check_pose(img, lmList, angleList, squat_status)

                if squat_status == "START":
                    #print("LOG - ", (lmList[13][2] + lmList[12][2]) / 2, (lmList[15][2] + lmList[14][2]) / 2)

                    if squat_updown == "UP":
                        if (lmList[13][2] + lmList[12][2]) / 2 >= (lmList[15][2] + lmList[14][2]) / 2:
                            squat_updown = "DOWN"
                    else:
                        if (lmList[13][2] + lmList[12][2]) / 2 < (lmList[15][2] + lmList[14][2]) / 2 and angleList[5] > 160:
                            squat_updown = "UP"
                            squat_count += 1
                            #print("LOG - ", squat_count)

            cv2.putText(img, user_name[0] + "_" + str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            cv2.putText(img, squat_status + "   " + squat_updown + "   " + str(squat_count) + "    " + acc_status, (450, 50),
                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            # 해당 이미지 출력
            #img = cv2.resize(img, dsize=(0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            fps_control = 0

        fps_control += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
