import cv2
import time
import redis
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from readDB import get_key_offset

CONFIDENCE_THRESHOLD = 0.6
TIME_THRESHOLD = 7
VELOCITY_THRESHOLD = 8
TARGET_MAX = 5
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
key_offset = get_key_offset()

model = YOLO('./runs/detect/train/weights/best.pt')
tracker = DeepSort(max_age=50)

#종로 2가 260, 240 / 540, 370 http://210.179.218.52:1935/live/151.stream/playlist.m3u8
#~~
#강남대로 120, 300 / 360, 420 http://cctvsec.ktict.co.kr/9999/7Hcw88TE2LcuSJfVUaH3ajtgPogS1WSjT8EJrzFRBxTowk1/AfBG5rtbj6Ck8sDf

url = ""  # cctv url
cap = cv2.VideoCapture(url)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

x1, y1 = 260, 240  # cctv 영상 중 원하는 구역만 자르기 frame = frame[y1:y2, x1:x2]
x2, y2 = 540, 370  # 필요한 부분만 잘라 확대하여 리소스 낭비 줄이고 검출에 용이하게 함

# Redis 연결 설정
redis_host = 'localhost'  # Redis 서버 주소
redis_port = 6379  # Redis 포트
redis_db = 0  # 사용할 Redis 데이터베이스 번호
redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=redis_db)

track_active = {}

while True:
    ret, frame = cap.read()

    #오류 발생 시 20초 후 캠 다시 시작
    if not ret:
        print('Cam Error')
        key_offset = get_key_offset()
        track_active.clear()
        cap.release()
        cv2.destroyAllWindows()
        time.sleep(20)
        cap = cv2.VideoCapture(url)
        continue

    frame = frame[y1:y2, x1:x2]
    frame = cv2.resize(frame, ((x2 - x1) * 2, (y2 - y1) * 2), interpolation=cv2.INTER_LINEAR)  # 영상 확대

    detection = model.predict(source=[frame], save=False)[0]
    results = []

    # ---------------------------------------------------------
    # 영상을 탐색하여 사람 발견 시 정보를 results에 추가
    # ---------------------------------------------------------
    for data in detection.boxes.data.tolist():
        confidence = float(data[4])
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])

        if class_id == 0 and len(track_active) < TARGET_MAX:    #최대 타깃 수를 정해 속도 저하 방지
            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

    tracks = tracker.update_tracks(results, frame=frame)

    # ---------------------------------------------------------
    # detection 으로 구한 대상에게 tracker 부여하며 구간 속력 계산
    # ---------------------------------------------------------
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()

        # 발견한 대상에 사각형과 id 표시
        xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        cv2.putText(frame, f"{track_id}", (xmin, ymin - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)

        # 현재 위치 설정. 횡단보도를 x 방향으로 이동하기 때문에 x 좌표만 판단
        current_location = (xmin + xmax) / 2

        # 처음 발견한 대상일 경우 [시작시간, 시작위치] 저장
        if track_id not in track_active:
            track_active[track_id] = [time.time(), current_location]

        # 기존에 있던 대상일 경우, (지금 시간 - 시작시간)을 구함
        else:
            time_interval = time.time() - track_active[track_id][0]
            cv2.putText(frame, f"%.2f" % (time_interval), (xmin + 20, ymin - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

            # 임계값(시간, 속도)이 넘어갈 경우, 적합한 대상이라고 판단하여 구간 속력을 구해 Redis에 저장
            if time_interval >= TIME_THRESHOLD:
                start_location = track_active[track_id][1]
                del track_active[track_id]

                distance = abs(current_location - start_location)
                average_velocity = round(distance / time_interval, 2)

                if average_velocity > VELOCITY_THRESHOLD: redis_client.set(track_id+key_offset, average_velocity)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()