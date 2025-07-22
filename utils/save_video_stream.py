import sys
from datetime import datetime
from threading import Thread
import cv2
import time
import multiprocessing as mp
from detectors.aruco.generate import get_streams, get_key_by_value


def show_progress_bar(duration, start_time, camera_name):
    while True:
        elapsed_time = time.time() - start_time
        progress = elapsed_time / duration
        bar_length = 40  # Длина прогресс-бара
        block = int(round(bar_length * progress))

        progress_bar = f"Камера {camera_name}: [{'#' * block}{'.' * (bar_length - block)}] {elapsed_time:.2f}/{duration:.2f} сек."
        sys.stdout.write(f"\r{progress_bar}")
        sys.stdout.flush()

        if elapsed_time >= duration:
            break

        time.sleep(0.1)  # Обновляем прогресс каждые 100 мс


def save_video_from_stream(stream_id, output_path, duration=600, camera_name='Camera'):
    cap = cv2.VideoCapture(stream_id)
    if not cap.isOpened():
        print(f"Ошибка: не удается открыть поток для {camera_name}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    start_time = time.time()

    progress_thread = Thread(target=show_progress_bar, args=(duration, start_time, camera_name))
    progress_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Не удалось захватить кадр для {camera_name}.")
            break

        show_frame_resized = cv2.resize(frame, (1920, 1080))
        out.write(show_frame_resized)

        if time.time() - start_time > duration:
            print(f"\nЗапись с камеры {camera_name} завершена.")
            break

    cap.release()
    out.release()
    print(f"\nВидео для {camera_name} сохранено в {output_path}")


def select_and_record_camera(stream_url, camera_name, duration=10):
    timestamp = datetime.now().strftime("%m_%d-%H_%M_%S")
    # Путь для сохранения видео
    output_path = f"clean_cameras/{camera_name}_video_{timestamp}.avi"

    # Запускаем запись с выбранного потока
    save_video_from_stream(stream_url, output_path, duration=duration, camera_name=camera_name)


if __name__ == '__main__':
    internal_streams, external_streams = get_streams()

    print("Available cameras:", ', '.join(internal_streams.keys()))

    while True:
        user_input = input("How many cameras? ")
        try:
            cameras_amount = int(user_input)
            break
        except ValueError:
            print("Invalid number of cameras")

    cameras = []  # имена камер
    all_streams = []

    for current_camera in range(cameras_amount):
        if cameras_amount == len(internal_streams.keys()):
            all_streams = internal_streams.values()
        else:
            current_camera_name = input(f"Camera #{current_camera + 1}: ")
            cameras.append(str(current_camera_name))
            all_streams.append(internal_streams[cameras[current_camera]])

    out_folder_name = 'videos_2810_office/'
    segment_duration = 5  # в секундах

    custom_dict_path = "data/custom_dicts/DICT_5X5_10000.yml"

    processes = []
    for stream in all_streams:
        current_camera_name = get_key_by_value(internal_streams, stream)
        process = mp.Process(target=select_and_record_camera,
                             args=(stream, current_camera_name, segment_duration))
        processes.append(process)

    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()
