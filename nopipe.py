import cv2
import time
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# 객체 탐지 모델 로드
def load_detection_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

# 객체 탐지 함수
def detect_objects(model, frame):
    transform = T.Compose([T.ToTensor()])
    frame_tensor = transform(frame)
    with torch.no_grad():
        start_time = time.time()
        predictions = model([frame_tensor])
        elapsed_time = time.time() - start_time
    return predictions[0], elapsed_time

# 영상 개선 함수
def enhance_frame(frame):
    # 각 픽셀의 밝기를 약간씩 높이는 방법을 사용하여 간단한 영상 개선 수행
    enhanced_frame = cv2.resize(frame,(480,480))
    enhanced_frame = frame.astype('float32') + 10  # 예시로 밝기를 10만큼 증가시킵니다.
    enhanced_frame = enhanced_frame.clip(0, 255).astype('uint8')  # 화소값을 0에서 255 사이로 잘라냅니다.
    time.sleep(1)
    return enhanced_frame

def main():
    # 객체 탐지 모델 로드
    detection_model = load_detection_model()

    video_path = '/home/ai/oms/car-detection.mp4'  # 동영상 파일 경로 설정
    cap = cv2.VideoCapture(video_path)  # 동영상 파일에서 비디오 캡처
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()

        # 영상 개선
        enhanced_frame = enhance_frame(frame)

        # 객체 탐지
        detection_results, detect_time = detect_objects(detection_model, enhanced_frame)

        total_time = time.time() - start_time

        # 개선된 영상과 객체 탐지 결과를 시각화하여 표시
        for box in detection_results['boxes']:
            box = [int(coord) for coord in box]
            cv2.rectangle(enhanced_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 각 프레임에 대한 처리 시간 출력
        print(f"Detection Time: {detect_time:.4f}s, Total Time: {total_time:.4f}s")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
