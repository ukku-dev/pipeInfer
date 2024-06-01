import cv2
import numpy as np
import threading
import queue
import time

def calculate_mean_variance(frame, output_queue):
    mean, stddev = cv2.meanStdDev(frame)
    variance = stddev**2
    print(f"Mean: {mean.flatten()}, Variance: {variance.flatten()}")
    flipped_frame = cv2.flip(frame, 1)  # 영상을 좌우 반전
    output_queue.put(flipped_frame)  # 결과를 큐에 넣음
    time.sleep(1)  # 1초 딜레이

def calculate_min_max(input_queue):
    flipped_frame = input_queue.get()  # 큐에서 반전된 프레임을 가져옴
    min_val = np.min(flipped_frame)
    max_val = np.max(flipped_frame)
    print(f"Min: {min_val}, Max: {max_val}")
    time.sleep(1)  # 1초 딜레이

def process_frame(frame):
    output_queue = queue.Queue()

    mean_variance_thread = threading.Thread(target=calculate_mean_variance, args=(frame, output_queue))
    min_max_thread = threading.Thread(target=calculate_min_max, args=(output_queue,))
    
    mean_variance_thread.start()
    min_max_thread.start()
    
    mean_variance_thread.join()
    min_max_thread.join()

if __name__ == "__main__":
    cap = cv2.VideoCapture('/home/ai/oms/car-detection.mp4')  # 비디오 파일 경로를 넣으세요

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_start_time = time.time()
        
        process_frame(frame)
        frame_count += 1

        frame_end_time = time.time()
        frame_processing_time = frame_end_time - frame_start_time
        print(f"Frame {frame_count} processed in {frame_processing_time} seconds")
        
        # 각 프레임을 처리한 후 1초 대기
        time.sleep(max(1 - frame_processing_time, 0))

        # 1초에 한 번씩 프레임을 읽기 위해 다음 프레임으로 이동
        cap.set(cv2.CAP_PROP_POS_MSEC, (frame_count * 1000))

    cap.release()
    cv2.destroyAllWindows()

    total_time = time.time() - start_time
    print(f"Processed {frame_count} frames in {total_time} seconds")
    print(f"FPS: {frame_count / total_time}")
