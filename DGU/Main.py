import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

CONFIDENCE_THRESHOLD = 0.65  # 신뢰 구간을 정하는 임계값. 참고 코드 기본값:0.6
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

model = YOLO('yolov8n.pt')
model = YOLO('./runs/detect/train/weights/best.pt')
tracker = DeepSort(max_age=50)

#강남대로
url = "http://cctvsec.ktict.co.kr/9999/7Hcw88TE2LcuSJfVUaH3av6VVB7e+jnwH4CIG87AqRctrfrPl7Q7R83SZuNsqt9V" # cctv url
cap = cv2.VideoCapture(url)

x1,y1 = 120,300     # cctv 영상 중 원하는 구역만 자르기 frame = frame[y1:y2, x1:x2]
x2,y2 = 360,420    # 필요한 부분만 잘라 확대하여 리소스 낭비 줄이고 검출에 용이하게 함

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

previous_tracks = {}

while True:

    ret, frame = cap.read()
    if not ret:
        print('Cam Error')
        break

    frame = frame[y1:y2, x1:x2]
    frame = cv2.resize(frame, ((x2-x1)*2, (y2-y1)*2), interpolation=cv2.INTER_LINEAR)   #영상 확대

    detection = model.predict(source=[frame], save=False)[0]
    results = []

    #-----------------------------------------------------
    # 영상을 탐색하여 사람 발견 시 정보를 results에 추가
    #-----------------------------------------------------
    for data in detection.boxes.data.tolist(): # data : [xmin, ymin, xmax, ymax, confidence_score, class_id]
        confidence = float(data[4])
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])

        if class_id == 0 : #검출한 대상이 사람일 경우만 필터링. 사람 class_id: 0
            results.append([[xmin, ymin, xmax-xmin, ymax-ymin], confidence, class_id])

    
    #-----------------------------------------------------
    # detection 으로 구한 대상에게 traker 부여하며 속력 계산
    #-----------------------------------------------------
    tracks = tracker.update_tracks(results, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()

        xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)

        # 검출한 대상의 속력 계산
        if track_id in previous_tracks:
            previous_box = previous_tracks[track_id]
            current_box = [xmin, ymin, xmax, ymax]

            # 현재 박스와 이전 박스 사이의 거리 계산
            distance = ((current_box[0] - previous_box[0])**2 + (current_box[1] - previous_box[1])**2)**0.5
            time_interval = 1  # 1프레임 간격을 가정
            velocity = distance / time_interval

            # 속력을 화면에 표시 (px/frame)
            cv2.putText(frame, f"Velocity: {velocity:.2f}", (xmin, ymin - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

        previous_tracks[track_id] = [xmin, ymin, xmax, ymax]

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'): # q 누르면 영상 종료
        break

cap.release()
cv2.destroyAllWindows()