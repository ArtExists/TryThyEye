import cv2
import mediapipe as mp
import numpy as np
from yolo_model import make_sunnyG

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(rot_mat[0, 0])
    sin = abs(rot_mat[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(image, rot_mat, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

cap = cv2.VideoCapture(0)
face = mp.solutions.face_mesh
facemesh = face.FaceMesh(max_num_faces=3)
flag = False

sunglasses = cv2.imread(make_sunnyG("images_test/satya.jpg"), cv2.IMREAD_UNCHANGED)
szzzz = 1.2

while True:
    suc, img = cap.read()
    if suc:
        text = '''Q: Quit\nM: Start Model\nN: Stop Model\nI: Increase Size\nD: Decrease Size'''
        ar=20
        for i in text.split('\n'):
            cv2.putText(img, i, org=(10, ar), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=0.5,
                    color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
            ar+=10

        h, w, _ = img.shape
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ress = facemesh.process(imgrgb)

        if ress.multi_face_landmarks:
            for lms in ress.multi_face_landmarks:
                left_eye = lms.landmark[127]
                right_eye = lms.landmark[356]
                nose = lms.landmark[6]

                left_eye_coords = (int(left_eye.x * w), int(left_eye.y * h))
                right_eye_coords = (int(right_eye.x * w), int(right_eye.y * h))
                nose_coords = (int(nose.x * w), int(nose.y * h))

                key = cv2.waitKey(1) & 0xFF
                if key == ord('n'):
                    flag = False
                if key == ord('m'):
                    flag = True
                if key == ord('i'):
                    szzzz += 0.01
                if key == ord('d'):
                    szzzz -= 0.01

                if flag:
                    sung_w = int(szzzz * (right_eye_coords[0] - left_eye_coords[0]))
                    sung_h = int((sung_w * sunglasses.shape[0] / sunglasses.shape[1]))
                    sung_resized = cv2.resize(sunglasses, (sung_w, sung_h), interpolation=cv2.INTER_AREA)

                    dx = right_eye_coords[0] - left_eye_coords[0]
                    dy = right_eye_coords[1] - left_eye_coords[1]
                    angle = -np.degrees(np.arctan2(dy, dx))  # flip sign

                    sung_rotated = rotate_image(sung_resized, angle)
                    sh, sw = sung_rotated.shape[:2]

                    cx = int((left_eye_coords[0] + right_eye_coords[0]) / 2)
                    cy = int((left_eye_coords[1] + right_eye_coords[1]) / 2 + (nose_coords[1] - (left_eye_coords[1] + right_eye_coords[1]) / 2) * 0.4)

                    x1 = cx - sw // 2
                    y1 = cy - sh // 2
                    x2 = x1 + sw
                    y2 = y1 + sh

                    x1_clip, y1_clip = max(0, x1), max(0, y1)
                    x2_clip, y2_clip = min(w, x2), min(h, y2)
                    sung_x1 = x1_clip - x1
                    sung_y1 = y1_clip - y1
                    sung_x2 = sung_x1 + (x2_clip - x1_clip)
                    sung_y2 = sung_y1 + (y2_clip - y1_clip)

                    alpha_s = sung_rotated[sung_y1:sung_y2, sung_x1:sung_x2, 3] / 255.0
                    alpha_l = 1.0 - alpha_s

                    for c in range(3):
                        img[y1_clip:y2_clip, x1_clip:x2_clip, c] = (
                            alpha_s * sung_rotated[sung_y1:sung_y2, sung_x1:sung_x2, c] +
                            alpha_l * img[y1_clip:y2_clip, x1_clip:x2_clip, c]
                        )

        cv2.imshow("Image", img)
    else:
        print("no succ")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
