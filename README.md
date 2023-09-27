# pedestrian_detection
Pedestrian detection and tracking with yolo v8, open cv

# subject
- 실시간 cctv 영상으로 횡단보도 보행자의 구간 속력을 구해 DB에 데이터 집계
- 집계된 데이터를 바탕으로 횡단보도의 청신호 유지시간 제안

# description
- open cctv api 를 통한 실시간 스트리밍 정보에서 횡단보도를 통행하는 보행자 검출
- Main.py : track_id를 바탕으로 이전 frame의 위치와 거리를 계산하여 보행자의 순간 속력 계산 (pixel / frame)
- test.py : 임의로 정한 n초 이상 지속적으로 검출되는 대상을 적합하다고 판단하여 n초 동안의 평균 속력을 DB에 전송 (pixel / s)
- train.py : Roboflow의 관련 이미지를 이용해 학습
- Redis : In Memory 방식으로 실시간으로 Id-Speed 값을 저장하기에 적합한 DB
- YOLOv8n : 실시간으로 영상 정보를 분석하기에 정확성이 비교적 낮더라도 빠른 속도를 보장하는 nano 버전 사용

# result
<p align="left">
<img src="https://github.com/hj78080/pedestrian_detection/assets/137899379/976b3aa2-64fe-4b8e-aa13-bcea2e1f90aa">
</p>
