import cv2
import mediapipe as mp
import math
import time
import numpy as np
from sklearn.linear_model import LogisticRegression

X_train = np.array([
    [1.0], [1.05], [0.95],
    [1.4], [1.5], [0.6], [0.55]
])
y_train = np.array([0,0,0,1,1,1,1])

model = LogisticRegression()
model.fit(X_train, y_train)

accuracy = model.score(X_train, y_train)
print("Model Accuracy:", accuracy)


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

last_alert_time = 0
alert_count = 0

def calc_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:

                mp_drawing.draw_landmarks(
                    image,
                    face,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_draw_styles.get_default_face_mesh_tesselation_style()
                )

                left_eye = face.landmark[33]
                right_eye = face.landmark[263]
                nose = face.landmark[1]

                eye_dist = calc_distance(left_eye, right_eye)
                nose_left = calc_distance(nose, left_eye)
                nose_right = calc_distance(nose, right_eye)
                ratio = nose_left / nose_right

                ai_pred = model.predict([[ratio]])[0]

                safe = 0.85 < ratio < 1.15
                cheating = (ai_pred == 1) and (not safe)

                if cheating:
                    current = time.time()
                    if current - last_alert_time > 2:
                        alert_count += 1
                        last_alert_time = current
                        print("Cheating Detected! (Student looked away)")
                        cv2.putText(image, "Cheating Detected! (Student looked away)", 
                                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9, (0, 0, 255), 3)

                h, w, _ = image.shape
                cv2.circle(image, (int(left_eye.x*w), int(left_eye.y*h)), 3, (0,255,0), -1)
                cv2.circle(image, (int(right_eye.x*w), int(right_eye.y*h)), 3, (0,255,0), -1)
                cv2.circle(image, (int(nose.x*w), int(nose.y*h)), 3, (0,0,255), -1)

        cv2.imshow("AI Cheating Detection", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print("\nTotal Alerts:", alert_count)
print("Program finished successfully.")