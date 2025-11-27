import cv2
from ultralytics import YOLO
import cvzone
import math

# 1. 모델 로드 (다운받은 best.pt 파일이 같은 폴더에 있어야 함)
# 만약 에러나면 경로를 수정하세요 (예: 'C:/Users/내이름/Desktop/ParkingSystem/best.pt')
model = YOLO('best.pt')

# 2. 클래스 이름 (Roboflow에서 설정한 이름, 보통 'car')
classNames = ['car']

# 3. 주차칸 좌표 설정 (가상 좌표)
# [x1, y1, x2, y2] - 왼쪽 위(x1, y1)와 오른쪽 아래(x2, y2) 좌표
# * 팁: 실행 후 화면을 보면서 내 웹캠 각도에 맞게 이 숫자를 수정해야 합니다!
parking_spots = [
    [50, 50, 250, 200],   # 1번 주차칸
    [300, 50, 500, 200],  # 2번 주차칸
    [550, 50, 750, 200]   # 3번 주차칸
]

# 4. 웹캠 열기 (0번은 보통 기본 웹캠)
cap = cv2.VideoCapture(1)
cap.set(3, 1280) # 해상도 너비
cap.set(4, 720)  # 해상도 높이

while True:
    success, img = cap.read()
    if not success: break

    # AI 추론 (이미지 분석)
    results = model(img, stream=True)
    
    # 감지된 차들의 중심점 좌표 저장 리스트
    car_centers = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # 신뢰도 50% 이상인 것만 인정
            conf = math.ceil((box.conf[0] * 100)) / 100
            if conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 차량 중심점 계산
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                car_centers.append((cx, cy))

                # (선택) 감지된 차에 보라색 박스 그리기
                # cvzone.cornerRect(img, (x1, y1, x2-x1, y2-y1), l=9, colorR=(255, 0, 255))

    # 5. 빈 자리 카운팅 로직
    occupied_count = 0
    for spot in parking_spots:
        sx1, sy1, sx2, sy2 = spot
        is_occupied = False
        
        # 차 중심점이 주차칸 안에 있는지 확인
        for cx, cy in car_centers:
            if sx1 < cx < sx2 and sy1 < cy < sy2:
                is_occupied = True
                break
        
        # 상태에 따라 색상 변경
        if is_occupied:
            color = (0, 0, 255) # 빨강 (주차됨)
            occupied_count += 1
        else:
            color = (0, 255, 0) # 초록 (비었음)
            
        # 주차칸 그리기
        cv2.rectangle(img, (sx1, sy1), (sx2, sy2), color, 3)

    # 6. 결과 텍스트 표시
    empty_spots = len(parking_spots) - occupied_count
    cvzone.putTextRect(img, f'Free: {empty_spots} / {len(parking_spots)}', (50, 50), scale=3, thickness=3, offset=10)

    cv2.imshow("Smart Parking System", img)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()