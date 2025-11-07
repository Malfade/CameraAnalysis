"""
Модуль для записи видео с камер.

Позволяет записывать видео поток с камеры в файл.
"""
import cv2
import os
import threading
from datetime import datetime
from typing import Optional, Dict


class VideoRecorder:
    """Класс для записи видео с камер."""
    
    def __init__(self, recordings_dir: str = "recordings", codec: str = "mp4v", fps: int = 30):
        """
        Инициализация видеорекордера.
        
        Args:
            recordings_dir: Директория для сохранения записей
            codec: Кодек для записи ('mp4v', 'XVID', 'MJPG')
            fps: Кадров в секунду
        """
        self.recordings_dir = recordings_dir
        self.codec = codec
        self.fps = fps
        os.makedirs(recordings_dir, exist_ok=True)
        
        # Словарь активных записей: {room_name: {'writer': VideoWriter, 'filepath': str}}
        self.active_recordings: Dict[str, Dict] = {}
        self.recording_lock = threading.Lock()
    
    def start_recording(self, room_name: str, frame_width: int, frame_height: int) -> Optional[str]:
        """
        Начать запись видео для комнаты.
        
        Args:
            room_name: Имя комнаты
            frame_width: Ширина кадра
            frame_height: Высота кадра
            
        Returns:
            Путь к файлу записи или None при ошибке
        """
        with self.recording_lock:
            # Если уже записывается - останавливаем предыдущую запись
            if room_name in self.active_recordings:
                self.stop_recording(room_name)
            
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"recording_{room_name}_{timestamp}.mp4"
                filepath = os.path.join(self.recordings_dir, filename)
                
                # Определяем кодек
                fourcc = cv2.VideoWriter_fourcc(*self.codec)
                
                # Создаем VideoWriter
                writer = cv2.VideoWriter(filepath, fourcc, self.fps, (frame_width, frame_height))
                
                if not writer.isOpened():
                    print(f"Ошибка: Не удалось открыть VideoWriter для {room_name}")
                    return None
                
                # Сохраняем информацию о записи
                self.active_recordings[room_name] = {
                    'writer': writer,
                    'filepath': filepath,
                    'start_time': datetime.now()
                }
                
                print(f"Начата запись для {room_name}: {filepath}")
                return filepath
            except Exception as e:
                print(f"Ошибка при начале записи для {room_name}: {e}")
                return None
    
    def add_frame(self, room_name: str, frame) -> bool:
        """
        Добавить кадр к записи.
        
        Args:
            room_name: Имя комнаты
            frame: Кадр изображения
            
        Returns:
            True если кадр добавлен, False если запись не активна
        """
        with self.recording_lock:
            if room_name not in self.active_recordings:
                return False
            
            try:
                writer = self.active_recordings[room_name]['writer']
                writer.write(frame)
                return True
            except Exception as e:
                print(f"Ошибка при записи кадра для {room_name}: {e}")
                return False
    
    def stop_recording(self, room_name: str) -> Optional[str]:
        """
        Остановить запись для комнаты.
        
        Args:
            room_name: Имя комнаты
            
        Returns:
            Путь к файлу записи или None
        """
        with self.recording_lock:
            if room_name not in self.active_recordings:
                return None
            
            try:
                recording_info = self.active_recordings[room_name]
                writer = recording_info['writer']
                filepath = recording_info['filepath']
                
                # Освобождаем ресурсы
                writer.release()
                
                # Удаляем из активных записей
                del self.active_recordings[room_name]
                
                duration = (datetime.now() - recording_info['start_time']).total_seconds()
                print(f"Остановлена запись для {room_name}: {filepath} (длительность: {duration:.1f}с)")
                
                return filepath
            except Exception as e:
                print(f"Ошибка при остановке записи для {room_name}: {e}")
                if room_name in self.active_recordings:
                    del self.active_recordings[room_name]
                return None
    
    def is_recording(self, room_name: str) -> bool:
        """Проверить, идет ли запись для комнаты."""
        with self.recording_lock:
            return room_name in self.active_recordings
    
    def get_recording_info(self, room_name: str) -> Optional[Dict]:
        """Получить информацию о текущей записи."""
        with self.recording_lock:
            if room_name not in self.active_recordings:
                return None
            
            recording_info = self.active_recordings[room_name]
            # Создаем копию без VideoWriter объекта (он не сериализуется в JSON)
            info = {
                'filepath': recording_info['filepath'],
                'start_time': recording_info['start_time'],
                'duration': (datetime.now() - recording_info['start_time']).total_seconds()
            }
            return info

