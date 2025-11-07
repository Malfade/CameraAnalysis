"""
Модуль для работы с камерами (USB и IP).

Этот модуль отвечает за захват видео с камер. Он поддерживает:
1. USB камеры (веб-камеры, встроенные камеры ноутбуков)
2. IP камеры через RTSP протокол (сетевые камеры)

Как это работает:
- Камера работает в отдельном потоке (thread)
- Постоянно читает кадры с камеры
- Сохраняет последний кадр в памяти
- Когда нужно получить кадр, просто возвращает сохраненный

Это позволяет не блокировать основной поток обработки - камера
постоянно обновляет кадр в фоне, а мы просто читаем его когда нужно.
"""
import cv2
import threading
import time
from typing import Optional, Callable
import numpy as np


class CameraStream:
    """
    Класс для захвата видео с камер.
    
    Этот класс:
    1. Подключается к камере (USB или IP)
    2. Запускает отдельный поток для чтения кадров
    3. Сохраняет последний кадр в памяти
    4. Предоставляет метод read() для получения текущего кадра
    """
    
    def __init__(self, camera_index, room_name: str):
        """
        Инициализация потока камеры.
        
        Args:
            camera_index: 
                - Если int (0, 1, 2...): индекс USB камеры
                  Обычно 0 - камера ноутбука, 1 - первая USB камера, и т.д.
                - Если str: RTSP URL для IP камеры
                  Например: "rtsp://admin:password@192.168.1.100:554/stream"
            room_name: Имя комнаты, к которой привязана камера (например, "Room1")
        """
        self.camera_index = camera_index
        self.room_name = room_name
        self.cap = None  # Объект VideoCapture от OpenCV (еще не инициализирован)
        self.frame = None  # Последний захваченный кадр
        self.running = False  # Флаг работы потока (True = работает, False = остановлен)
        self.lock = threading.Lock()  # Lock для потокобезопасного доступа к frame
        self.thread = None  # Поток, который будет читать кадры
        
        # Метрики для мониторинга
        self.frame_count = 0  # Счетчик кадров
        self.fps_start_time = time.time()  # Время начала для расчета FPS
        self.current_fps = 0.0  # Текущий FPS
        self.last_frame_time = None  # Время последнего кадра
        self.frame_width = 0  # Ширина кадра
        self.frame_height = 0  # Высота кадра
        self.error_count = 0  # Счетчик ошибок
        self.last_error = None  # Последняя ошибка
        
    def start(self):
        """
        Запустить поток камеры.
        
        Эта функция:
        1. Открывает камеру (USB или IP)
        2. Настраивает параметры (разрешение, FPS)
        3. Запускает поток для чтения кадров
        4. Ждет, пока появится первый кадр
        """
        # Если поток уже запущен, ничего не делаем
        if self.running:
            return
        
        # ============================================
        # ОТКРЫТИЕ КАМЕРЫ
        # ============================================
        # Определяем, USB камера или IP камера
        if isinstance(self.camera_index, str):
            # Это RTSP URL для IP камеры
            # cv2.CAP_FFMPEG - используем FFmpeg для обработки RTSP потока
            # FFmpeg - это библиотека для работы с видео/аудио потоками
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_FFMPEG)
        else:
            # Это USB камера (индекс - число)
            # OpenCV автоматически найдет камеру с этим индексом
            self.cap = cv2.VideoCapture(self.camera_index)
        
        # Проверяем, успешно ли открылась камера
        if not self.cap.isOpened():
            # Если камера не открылась, выбрасываем ошибку
            raise RuntimeError(f"Не удалось открыть камеру {self.camera_index} для {self.room_name}")
        
        # ============================================
        # НАСТРОЙКА ПАРАМЕТРОВ КАМЕРЫ
        # ============================================
        # Устанавливаем разрешение кадра (640x480 - стандартное разрешение)
        # cv2.CAP_PROP_FRAME_WIDTH - свойство ширины кадра
        # cv2.CAP_PROP_FRAME_HEIGHT - свойство высоты кадра
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Получаем реальное разрешение камеры
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Устанавливаем FPS (кадров в секунду)
        # 30 FPS - стандартная частота для видео
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # ============================================
        # ЗАПУСК ПОТОКА ДЛЯ ЧТЕНИЯ КАДРОВ
        # ============================================
        self.running = True  # Устанавливаем флаг работы
        
        # Создаем поток для чтения кадров
        # threading.Thread создает новый поток выполнения
        # target=self._update_frame - функция, которая будет выполняться в потоке
        # daemon=True - поток будет автоматически завершен при завершении главного процесса
        self.thread = threading.Thread(target=self._update_frame, daemon=True)
        self.thread.start()  # Запускаем поток
        
        # Ждем немного, чтобы поток успел захватить первый кадр
        # Это важно, чтобы при первом вызове read() кадр уже был доступен
        time.sleep(0.5)
    
    def _update_frame(self):
        """
        Обновление кадра в отдельном потоке.
        
        Эта функция работает в бесконечном цикле в отдельном потоке:
        1. Читает кадр с камеры
        2. Сохраняет его в self.frame
        3. Повторяет процесс постоянно
        
        Благодаря этому, последний кадр всегда доступен через метод read(),
        и мы не блокируем основной поток обработки.
        """
        # Бесконечный цикл - работает пока self.running = True
        while self.running:
            # Читаем кадр с камеры
            # cap.read() возвращает два значения:
            # - ret: True если кадр успешно прочитан, False если ошибка
            # - frame: numpy array с изображением кадра (если ret=True)
            ret, frame = self.cap.read()
            
            if ret:
                # Кадр успешно прочитан
                # Используем lock для потокобезопасности
                # Без lock может возникнуть race condition: один поток читает,
                # другой записывает одновременно
                with self.lock:
                    # Сохраняем копию кадра (frame.copy() нужен, чтобы не изменять
                    # оригинальный frame, который может использоваться OpenCV)
                    self.frame = frame.copy()
                
                # Обновляем метрики
                self.frame_count += 1
                current_time = time.time()
                self.last_frame_time = current_time
                
                # Вычисляем FPS каждую секунду
                elapsed = current_time - self.fps_start_time
                if elapsed >= 1.0:
                    self.current_fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.fps_start_time = current_time
                
                # Обновляем разрешение если изменилось
                if frame is not None:
                    h, w = frame.shape[:2]
                    self.frame_width = w
                    self.frame_height = h
            else:
                # Не удалось прочитать кадр (камера отключена, ошибка сети и т.д.)
                self.error_count += 1
                self.last_error = "Failed to read frame"
                print(f"Предупреждение: Не удалось прочитать кадр с камеры {self.room_name}")
                # Ждем немного перед следующей попыткой
                time.sleep(0.1)
    
    def read(self) -> Optional[np.ndarray]:
        """
        Получить текущий кадр.
        
        Этот метод возвращает последний захваченный кадр.
        Он не блокирует выполнение - просто возвращает то, что уже есть в памяти.
        
        Returns:
            Кадр изображения в формате numpy array (BGR, 3 канала)
            или None, если кадр еще не был захвачен
        """
        # Используем lock для потокобезопасности
        with self.lock:
            if self.frame is not None:
                # Возвращаем копию кадра (чтобы не изменять оригинал)
                return self.frame.copy()
            return None
    
    def get_status(self) -> dict:
        """
        Получить статус камеры.
        
        Returns:
            Словарь с информацией о состоянии камеры
        """
        is_opened = self.cap is not None and self.cap.isOpened() if self.cap else False
        status = "active" if (is_opened and self.running and self.frame is not None) else "inactive"
        
        return {
            "room_name": self.room_name,
            "status": status,
            "fps": round(self.current_fps, 2),
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "running": self.running,
            "last_frame_time": self.last_frame_time
        }
    
    def stop(self):
        """
        Остановить поток камеры.
        
        Эта функция:
        1. Устанавливает флаг self.running = False (останавливает цикл)
        2. Ждет завершения потока (до 2 секунд)
        3. Освобождает камеру (release)
        
        Важно вызывать эту функцию перед завершением программы,
        чтобы корректно освободить ресурсы камеры.
        """
        # Останавливаем цикл в _update_frame
        self.running = False
        
        # Ждем завершения потока (timeout=2.0 секунды)
        # thread.join() блокирует выполнение, пока поток не завершится
        if self.thread:
            self.thread.join(timeout=2.0)
        
        # Освобождаем камеру
        # cap.release() закрывает соединение с камерой
        if self.cap:
            self.cap.release()
    
    def is_opened(self) -> bool:
        """
        Проверить, открыта ли камера.
        
        Returns:
            True если камера открыта и работает, False иначе
        """
        # Проверяем, что объект VideoCapture создан и камера открыта
        return self.cap is not None and self.cap.isOpened()