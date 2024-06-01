import threading
import queue
import time

def producer(q, n):
    for i in range(n):
        item = f"item {i}"
        q.put(item)
        print(f"Produced: {item}")
        time.sleep(1)

def consumer(q):
    while True:
        item = q.get()
        if item is None:  # 종료 신호
            break
        print(f"Consumed: {item}")
        q.task_done()

if __name__ == "__main__":
    q = queue.Queue()
    num_items = 5

    producer_thread = threading.Thread(target=producer, args=(q, num_items))
    consumer_thread = threading.Thread(target=consumer, args=(q,))

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    q.put(None)  # 소비자에게 종료 신호 보내기
    consumer_thread.join()