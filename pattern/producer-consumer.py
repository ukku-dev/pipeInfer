import cv2
import numpy as np
import threading
import queue
import time

def capture_frames(video_path, frame_queue, stop_event):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put((frame, time.time()))
        time.sleep(1)  # 1초마다 프레임을 캡처
    
    cap.release()
    frame_queue.put((None, None))  # 종료 신호

def calculate_mean_variance(frame_queue, flip_queue, stop_event):
    while not stop_event.is_set():
        frame, capture_time = frame_queue.get()
        if frame is None:
            flip_queue.put((None, None))
            break
        start_time = time.time()
        mean, stddev = cv2.meanStdDev(frame)
        variance = stddev**2
        print(f"Mean: {mean.flatten()}, Variance: {variance.flatten()}")
        flipped_frame = cv2.flip(frame, 1)  # 영상을 좌우 반전
        processing_time = time.time() - start_time
        flip_queue.put((flipped_frame, capture_time, processing_time))
        time.sleep(1)  # 1초 딜레이

def calculate_min_max(flip_queue, stop_event):
    while not stop_event.is_set():
        flipped_frame, capture_time, mean_variance_time = flip_queue.get()
        if flipped_frame is None:
            break
        start_time = time.time()
        min_val = np.min(flipped_frame)
        max_val = np.max(flipped_frame)
        processing_time = time.time() - start_time
        total_processing_time = (time.time() - capture_time)
        print(f"Min: {min_val}, Max: {max_val}")
        print(f"Frame processed in {total_processing_time} seconds")
        time.sleep(1)  # 1초 딜레이

if __name__ == "__main__":
    video_path = '/home/ai/oms/car-detection.mp4'  # 비디오 파일 경로를 넣으세요
    frame_queue = queue.Queue()
    flip_queue = queue.Queue()
    stop_event = threading.Event()

    capture_thread = threading.Thread(target=capture_frames, args=(video_path, frame_queue, stop_event))
    mean_variance_thread = threading.Thread(target=calculate_mean_variance, args=(frame_queue, flip_queue, stop_event))
    min_max_thread = threading.Thread(target=calculate_min_max, args=(flip_queue, stop_event))

    capture_thread.start()
    mean_variance_thread.start()
    min_max_thread.start()

    try:
        while capture_thread.is_alive():
            time.sleep(1)  # 메인 스레드를 계속 돌려서 종료를 방지
    except KeyboardInterrupt:
        stop_event.set()

    capture_thread.join()
    mean_variance_thread.join()
    min_max_thread.join()

    print("Processing complete.")
