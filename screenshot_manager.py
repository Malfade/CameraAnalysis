"""
Модуль для сохранения скриншотов при событиях.

Сохраняет кадры при:
- Входе в комнату
- Выходе из комнаты
- Перемещении между комнатами
"""
import cv2
import os
from datetime import datetime
from typing import Optional


class ScreenshotManager:
    """Класс для управления скриншотами."""
    
    def __init__(self, screenshots_dir: str = "screenshots", auto_enabled: bool = True, jpeg_quality: int = 85):
        """
        Инициализация менеджера скриншотов.
        
        Args:
            screenshots_dir: Директория для сохранения скриншотов
            auto_enabled: Включены ли автоматические скриншоты
            jpeg_quality: Качество JPEG (1-100)
        """
        self.screenshots_dir = screenshots_dir
        self.auto_enabled = auto_enabled
        self.jpeg_quality = jpeg_quality
        # Создаем директорию, если её нет
        os.makedirs(screenshots_dir, exist_ok=True)
    
    def save_screenshot(self, frame, person_id: str, event_type: str, 
                       room_name: str, additional_info: str = "") -> Optional[str]:
        """
        Сохранить скриншот при событии.
        
        Args:
            frame: Кадр изображения (numpy array)
            person_id: ID человека (p1, p2, ...)
            event_type: Тип события ('enter', 'exit', 'move')
            room_name: Имя комнаты
            additional_info: Дополнительная информация (например, для 'move' - "Room1->Room2")
            
        Returns:
            Путь к сохраненному файлу или None при ошибке
        """
        try:
            # Формируем имя файла
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{person_id}_{event_type}_{room_name}_{timestamp}.jpg"
            
            # Если есть дополнительная информация, добавляем её
            if additional_info:
                # Заменяем спецсимволы на безопасные
                safe_info = additional_info.replace("->", "_to_").replace(" ", "_")
                filename = f"{person_id}_{event_type}_{safe_info}_{timestamp}.jpg"
            
            filepath = os.path.join(self.screenshots_dir, filename)
            
            # Сохраняем кадр
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            
            return filepath
        except Exception as e:
            print(f"Ошибка при сохранении скриншота: {e}")
            return None
    
    def save_manual_screenshot(self, frame, room_name: str, label: str = "manual") -> Optional[str]:
        """
        Сохранить скриншот вручную (не при событии).
        
        Args:
            frame: Кадр изображения (numpy array)
            room_name: Имя комнаты
            label: Метка для скриншота (по умолчанию "manual")
            
        Returns:
            Путь к сохраненному файлу или None при ошибке
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{label}_{room_name}_{timestamp}.jpg"
            filepath = os.path.join(self.screenshots_dir, filename)
            
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            
            return filepath
        except Exception as e:
            print(f"Ошибка при сохранении ручного скриншота: {e}")
            return None
    
    def save_enter_screenshot(self, frame, person_id: str, room_name: str) -> Optional[str]:
        """Сохранить скриншот при входе в комнату."""
        return self.save_screenshot(frame, person_id, "enter", room_name)
    
    def save_exit_screenshot(self, frame, person_id: str, room_name: str) -> Optional[str]:
        """Сохранить скриншот при выходе из комнаты."""
        return self.save_screenshot(frame, person_id, "exit", room_name)
    
    def save_move_screenshot(self, frame, person_id: str, from_room: str, to_room: str) -> Optional[str]:
        """Сохранить скриншот при перемещении между комнатами."""
        move_info = f"{from_room}->{to_room}"
        return self.save_screenshot(frame, person_id, "move", to_room, move_info)
