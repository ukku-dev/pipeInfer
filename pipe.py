import threading
from queue import Queue
import time
import cv2
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
#from torchvision.models import srresnet

# 영상 개선(Super Resolution) 함수
# def enhance_frame(model, frame):
#     transform = T.Compose([
#         T.ToTensor(),
#         T.Resize((frame.shape[0] // 4, frame.shape[1] // 4)),
#         T.Resize((frame.shape[0], frame.shape[1]), interpolation=T.InterpolationMode.BICUBIC),
#         T.ToPILImage(),
#         T.ToTensor()
#     ])
#     frame_tensor = transform(frame).unsqueeze(0)
#     with torch.no_grad():
#         enhanced_tensor = model(frame_tensor)
#     enhanced_frame = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
#     enhanced_frame = (enhanced_frame * 255).astype('uint8')
#     return enhanced_frame
# 영상 개선 함수
def enhance_frame(frame):
    # 각 픽셀의 밝기를 약간씩 높이는 방법을 사용하여 간단한 영상 개선 수행
    enhanced_frame = cv2.resize(frame,(480,480))
    enhanced_frame = frame.astype('float32') + 10  # 예시로 밝기를 10만큼 증가시킵니다.
    enhanced_frame = enhanced_frame.clip(0, 255).astype('uint8')  # 화소값을 0에서 255 사이로 잘라냅니다.
    time.sleep(1)
    return enhanced_frame

# 객체 탐지 함수
def detect_objects(model, frame):
    transform = T.Compose([T.ToTensor()])
    frame_tensor = transform(frame)
    with torch.no_grad():
        predictions = model([frame_tensor])
    return predictions[0]

def enhance_video(frame_queue, enhanced_queue):
    while True:
        frame = frame_queue.get()
        
        #print(f'enhance_video, {enhanced_queue.qsize()}')
        if frame is None:
            break
        start_time = time.time()
        enhanced_frame = enhance_frame(frame)
        enhance_time = time.time() - start_time
        enhanced_queue.put((enhanced_frame, enhance_time))

def detect_objects_thread(detect_model, enhanced_queue, result_queue):
    while True:
        enhanced_data = enhanced_queue.get()
        #print(f'detect_objects_thread, {enhanced_queue.qsize()}')
        if enhanced_data is None:
            result_queue.put(None)
            #print('break')
            continue
        
        enhanced_frame, enhance_time = enhanced_data
        start_time = time.time()
        detection_results = detect_objects(detect_model, enhanced_frame)
        detect_time = time.time() - start_time
        result_queue.put((detection_results, enhance_time, detect_time))
        #print(f'result_queue, {result_queue.qsize()}')

def main():
    frame_queue = Queue()
    enhanced_queue = Queue()
    result_queue = Queue()

    # Super Resolution 모델 로드
    # enhance_model = srresnet(pretrained=True)
    # enhance_model.eval()

    # Faster R-CNN 모델 로드
    detect_model = fasterrcnn_resnet50_fpn(pretrained=True)
    detect_model.eval()

    enhance_thread = threading.Thread(target=enhance_video, args=(frame_queue, enhanced_queue))
    detect_thread = threading.Thread(target=detect_objects_thread, args=(detect_model, enhanced_queue, result_queue))

    enhance_thread.start()
    detect_thread.start()

    video_path = '/home/ai/oms/car-detection.mp4'  # 동영상 파일 경로 설정
    cap = cv2.VideoCapture(video_path)  # 동영상 파일에서 비디오 캡처
    tpf= 0
    tpf_pre = 0
    while True:
        ret, frame = cap.read()
        if ret:
            frame_queue.put(frame)
            #break

        
        if not result_queue.empty():
            
            print(f'{tpf-tpf_pre} time per frame')
            detection_results, enhance_time, detect_time = result_queue.get()
            total_time = enhance_time + detect_time

            # 각 프레임에 대한 처리 시간 출력
            print(f"Enhancement Time: {enhance_time:.4f}s, Detection Time: {detect_time:.4f}s, Total Time: {total_time:.4f}s")

            for box in detection_results['boxes']:
                x1, y1, x2, y2 = box.int().tolist()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            tpf_pre = tpf
            tpf = time.time()
            #cv2.imshow('Object Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 종료를 알리기 위해 None을 큐에 추가
    frame_queue.put(None)
    enhanced_queue.put(None)
    
    enhance_thread.join()
    detect_thread.join()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
