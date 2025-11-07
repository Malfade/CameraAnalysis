"""
Модуль для управления комнатами и отслеживания перемещений людей.

Этот модуль - это "мозг" системы. Он отслеживает:
1. Кто находится в какой комнате в данный момент
2. Когда люди появляются и исчезают
3. Перемещения людей между комнатами

Логика перемещений:
- Если человек исчез из Room1 и появился в Room2 в течение 1-7 секунд -
  это считается перемещением
- Если человек просто исчез и не появился в другой комнате -
  он удаляется из системы через некоторое время

Как это работает:
1. Каждая камера постоянно сообщает, кого она видит (список ID)
2. RoomManager сравнивает новый список со старым:
   - Если появился новый ID - проверяет, не перемещение ли это
   - Если ID исчез - добавляет в список "исчезнувших"
3. Если человек исчез из одной комнаты и появился в другой -
  записывается событие перемещения в базу данных
"""
import time
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
from database import Database


class RoomManager:
    """
    Класс для управления людьми в комнатах и отслеживания перемещений.
    
    Этот класс хранит состояние системы:
    - Кто сейчас в какой комнате
    - Кто недавно исчез и откуда
    - Когда происходили перемещения
    """
    
    def __init__(self, database: Database, movement_window: float = 7.0, screenshot_manager=None):
        """
        Инициализация менеджера комнат.
        
        Args:
            database: Экземпляр базы данных для сохранения информации
            movement_window: Окно времени для определения перемещения (секунды)
                           Если человек исчез и появился в другой комнате
                           в течение этого времени - это перемещение
            screenshot_manager: Менеджер скриншотов для сохранения кадров при событиях (опционально)
        """
        self.db = database
        self.movement_window = movement_window
        self.screenshot_manager = screenshot_manager
        # Храним последние кадры для каждой комнаты (для скриншотов)
        self.last_frames = {}  # {room_name: frame}
        
        # ============================================
        # СТРУКТУРЫ ДАННЫХ ДЛЯ ОТСЛЕЖИВАНИЯ
        # ============================================
        
        # Текущее состояние: кто сейчас в какой комнате
        # Формат: {room_name: {person_id: timestamp}}
        # Пример: {"Room1": {"p1": 1234567890.5, "p2": 1234567891.2}}
        # Это означает, что в Room1 сейчас 2 человека (p1 и p2)
        # и время их последнего обнаружения
        self.current_people: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Люди, которые исчезли (но могут появиться в другой комнате)
        # Формат: {person_id: (room_name, timestamp)}
        # Пример: {"p3": ("Room1", 1234567890.5)}
        # Это означает, что p3 исчез из Room1 в момент времени 1234567890.5
        # и мы ждем, не появится ли он в другой комнате
        self.disappeared_people: Dict[str, Tuple[str, float]] = {}
        
        # Lock для потокобезопасности
        # Несколько камер могут одновременно обновлять информацию
        # Lock гарантирует, что данные не будут испорчены
        import threading
        self.lock = threading.Lock()
    
    def update_room(self, room_name: str, detected_person_ids: List[str], frame=None) -> List[Dict]:
        """
        Обновить список людей в комнате.
        
        Это главная функция класса. Она вызывается для каждой камеры
        каждый кадр и обновляет информацию о том, кто находится в комнате.
        
        Алгоритм работы:
        1. Сравниваем новый список людей со старым
        2. Находим новых людей (появились) и исчезнувших
        3. Для новых - проверяем, не перемещение ли это из другой комнаты
        4. Для исчезнувших - добавляем в список исчезнувших
        5. Обновляем текущее состояние комнаты
        
        Args:
            room_name: Имя комнаты (например, "Room1")
            detected_person_ids: Список ID людей, обнаруженных в этой комнате
                                Пример: ["p1", "p2", "p4"]
            frame: Кадр изображения для сохранения скриншотов (опционально)
        """
        # Сохраняем кадр для возможных скриншотов
        if frame is not None:
            self.last_frames[room_name] = frame
        
        # Используем lock для потокобезопасности
        events = []  # Список событий для возврата (для группового анализа)
        
        with self.lock:
            current_time = time.time()  # Текущее время (Unix timestamp)
            
            # ============================================
            # ЭТАП 1: ПОДГОТОВКА ДАННЫХ
            # ============================================
            # Преобразуем список в множество для быстрого сравнения
            # Множество (set) позволяет быстро находить различия
            detected_set = set(detected_person_ids)
            
            # Получаем множество людей, которые были в комнате ранее
            current_in_room = set(self.current_people[room_name].keys())
            
            # ============================================
            # ЭТАП 2: НАХОДИМ ИЗМЕНЕНИЯ
            # ============================================
            # Новые люди - те, кого раньше не было, а теперь есть
            # Пример: было ["p1", "p2"], стало ["p1", "p2", "p3"]
            # new_people = {"p3"}
            new_people = detected_set - current_in_room
            
            # Исчезнувшие люди - те, кто был раньше, а теперь нет
            # Пример: было ["p1", "p2"], стало ["p1"]
            # disappeared = {"p2"}
            disappeared = current_in_room - detected_set
            
            # ============================================
            # ЭТАП 3: ОБНОВЛЯЕМ ТЕКУЩИХ ЛЮДЕЙ
            # ============================================
            # Обновляем информацию о всех обнаруженных людях
            for person_id in detected_set:
                # Проверяем, новый ли это человек в комнате
                is_new = person_id not in self.current_people[room_name]
                
                # Обновляем время последнего обнаружения
                self.current_people[room_name][person_id] = current_time
                # Обновляем информацию в базе данных
                self.db.update_person_location(person_id, room_name)
                
                # Если это новый человек - фиксируем вход в комнату
                if is_new:
                    self.db.start_room_visit(person_id, room_name)
                    # Сохраняем скриншот при входе (только если автоматические скриншоты включены)
                    if (self.screenshot_manager and 
                        self.screenshot_manager.auto_enabled and 
                        room_name in self.last_frames):
                        self.screenshot_manager.save_enter_screenshot(
                            self.last_frames[room_name], person_id, room_name
                        )
            
            # ============================================
            # ЭТАП 4: ОБРАБАТЫВАЕМ ИСЧЕЗНУВШИХ
            # ============================================
            # Если человек исчез из кадра, он может:
            # 1. Уйти из комнаты (переместиться в другую комнату)
            # 2. Просто выйти за пределы кадра (но остаться в комнате)
            # 3. Уйти из здания
            
            for person_id in disappeared:
                # Получаем время, когда человек был замечен в последний раз
                disappear_time = self.current_people[room_name].get(person_id, current_time)
                
                # Добавляем в список исчезнувших
                # Это нужно, чтобы потом проверить, не появился ли он в другой комнате
                self.disappeared_people[person_id] = (room_name, disappear_time)
                
                # Удаляем из текущего списка людей в комнате
                del self.current_people[room_name][person_id]
                
                # Фиксируем выход из комнаты
                self.db.end_room_visit(person_id, room_name)
                # Сохраняем скриншот при выходе
                if self.screenshot_manager and room_name in self.last_frames:
                    self.screenshot_manager.save_exit_screenshot(
                        self.last_frames[room_name], person_id, room_name
                    )
            
            # ============================================
            # ЭТАП 5: ОБРАБАТЫВАЕМ НОВЫХ ЛЮДЕЙ
            # ============================================
            # Если появился новый человек, он может быть:
            # 1. Новым посетителем (впервые появился)
            # 2. Человеком, который переместился из другой комнаты
            # 3. Человеком, который был в другой комнате одновременно
            
            for person_id in new_people:
                # Сначала проверяем, не находится ли человек уже в другой комнате
                # Это может произойти, если человек одновременно виден на двух камерах
                # (например, в дверном проеме между комнатами)
                old_room = None
                
                # Ищем во всех других комнатах
                for other_room, people in self.current_people.items():
                    if other_room != room_name and person_id in people:
                        # Нашли! Человек уже в другой комнате
                        old_room = other_room
                        # Удаляем его из старой комнаты
                        del self.current_people[other_room][person_id]
                        break
                
                # Если человек был в другой комнате, это перемещение
                if old_room:
                    # Записываем перемещение в базу данных
                    self.db.add_movement(person_id, old_room, room_name)
                    print(f"Перемещение: {person_id} из {old_room} в {room_name}")
                    # Сохраняем скриншот при перемещении (только если автоматические скриншоты включены)
                    if (self.screenshot_manager and 
                        self.screenshot_manager.auto_enabled and 
                        room_name in self.last_frames):
                        self.screenshot_manager.save_move_screenshot(
                            self.last_frames[room_name], person_id, old_room, room_name
                        )
                    # Добавляем событие для группового анализа
                    events.append({"type": "move", "person_id": person_id, "from_room": old_room, "to_room": room_name, "timestamp": current_time})
                else:
                    # Человек не был в другой комнате
                    # Проверяем, не был ли он в списке исчезнувших
                    # (тогда это перемещение из комнаты, где он исчез)
                    self._check_movement(person_id, room_name, current_time, events)
        
        # Возвращаем список событий для дальнейшей обработки
        return events
    
    def _check_movement(self, person_id: str, new_room: str, current_time: float, events: List[Dict]):
        """
        Проверить, является ли появление человека перемещением из другой комнаты.
        
        Эта функция проверяет список исчезнувших людей. Если человек недавно
        исчез из одной комнаты и появился в другой - это перемещение.
        
        Алгоритм:
        1. Проверяем, есть ли человек в списке исчезнувших
        2. Если есть, проверяем время между исчезновением и появлением
        3. Если время в допустимом окне (1-7 секунд) - это перемещение
        4. Записываем перемещение в базу данных
        
        Args:
            person_id: ID человека (p1, p2, p3...)
            new_room: Новая комната, где человек появился
            current_time: Текущее время (Unix timestamp)
        """
        # Проверяем, есть ли человек в списке исчезнувших
        if person_id in self.disappeared_people:
            # Получаем информацию: из какой комнаты исчез и когда
            old_room, disappear_time = self.disappeared_people[person_id]
            
            # Вычисляем время между исчезновением и появлением
            time_diff = current_time - disappear_time
            
            # Проверяем, попадает ли перемещение в временное окно
            # 1.0 секунда - минимальное время (чтобы не считать быстрое исчезновение/появление)
            # movement_window (обычно 7.0) - максимальное время
            # Если человек исчез и появился через 10 секунд - это не перемещение,
            # он мог уйти и прийти обратно
            if 1.0 <= time_diff <= self.movement_window:
                # Это перемещение между комнатами!
                # Записываем в базу данных
                self.db.add_movement(person_id, old_room, new_room)
                print(f"Перемещение: {person_id} из {old_room} в {new_room} (время: {time_diff:.2f}с)")
                # Сохраняем скриншот при перемещении (только если автоматические скриншоты включены)
                if (self.screenshot_manager and 
                    self.screenshot_manager.auto_enabled and 
                    new_room in self.last_frames):
                    self.screenshot_manager.save_move_screenshot(
                        self.last_frames[new_room], person_id, old_room, new_room
                    )
                # Добавляем событие для группового анализа
                events.append({"type": "move", "person_id": person_id, "from_room": old_room, "to_room": new_room, "timestamp": current_time})
            
            # Удаляем из списка исчезнувших (он больше не исчезнувший, он найден)
            del self.disappeared_people[person_id]
        else:
            # Человек не был в списке исчезнувших
            # Это означает, что он:
            # 1. Впервые появился в системе
            # 2. Или исчез слишком давно (больше movement_window секунд)
            # В этом случае мы не записываем перемещение
            pass
    
    def get_all_rooms_status(self) -> Dict[str, Dict]:
        """
        Получить статус всех комнат для WebSocket.
        
        Returns:
            Словарь с информацией о комнатах
        """
        with self.lock:
            result = {}
            for room_name, people in self.current_people.items():
                result[room_name] = {
                    "count": len(people),
                    "persons": sorted(list(people.keys()))
                }
            return result
    
    def cleanup_old_disappeared(self, max_age: float = 10.0):
        """
        Очистить старые записи об исчезнувших людях.
        
        Если человек исчез и не появился в другой комнате в течение max_age секунд,
        он удаляется из системы. Это нужно, чтобы не засорять память.
        
        Эта функция вызывается периодически (каждые 5 секунд) в отдельном потоке.
        
        Args:
            max_age: Максимальный возраст записи в секундах (по умолчанию 10)
                    Если человек исчез более max_age секунд назад и не появился -
                    удаляем его из системы
        """
        with self.lock:
            current_time = time.time()
            to_remove = []  # Список людей для удаления
            
            # Проходим по всем исчезнувшим людям
            for person_id, (room_name, disappear_time) in self.disappeared_people.items():
                # Вычисляем возраст записи
                age = current_time - disappear_time
                
                # Если запись слишком старая, добавляем в список на удаление
                if age > max_age:
                    to_remove.append(person_id)
                    # Удаляем из базы данных
                    self.db.remove_person_from_room(person_id)
            
            # Удаляем старые записи
            for person_id in to_remove:
                del self.disappeared_people[person_id]
    
    def get_room_status(self) -> Dict[str, Dict]:
        """
        Получить текущий статус всех комнат.
        
        Возвращает информацию о том, сколько людей в каждой комнате
        и их ID. Используется для отображения на веб-панели.
        
        Returns:
            Словарь в формате:
            {
                "Room1": {"count": 3, "persons": ["p1", "p2", "p4"]},
                "Room2": {"count": 1, "persons": ["p3"]}
            }
        """
        with self.lock:
            result = {}
            # Проходим по всем комнатам с людьми
            for room_name, people in self.current_people.items():
                result[room_name] = {
                    "count": len(people),  # Количество людей
                    "persons": sorted(people.keys())  # Список ID (отсортированный)
                }
            return result
    
    def get_all_rooms_status(self) -> Dict[str, Dict]:
        """
        Получить статус всех комнат, включая пустые.
        
        Эта функция возвращает статус всех комнат из базы данных,
        даже если в них сейчас никого нет. Используется для отображения
        на веб-панели, чтобы показать все комнаты.
        
        Returns:
            Словарь со статусом всех комнат:
            {
                "Room1": {"count": 3, "persons": ["p1", "p2", "p4"]},
                "Room2": {"count": 0, "persons": []},
                "Room3": {"count": 1, "persons": ["p5"]}
            }
        """
        # Получаем список всех комнат из базы данных
        rooms = self.db.get_rooms()
        
        # Получаем статус комнат с людьми
        room_status = self.get_room_status()
        
        # Формируем результат со всеми комнатами
        result = {}
        for room in rooms:
            room_name = room['name']
            if room_name in room_status:
                # Комната есть в статусе - копируем данные
                result[room_name] = room_status[room_name]
            else:
                # Комнаты нет в статусе - значит, она пустая
                result[room_name] = {"count": 0, "persons": []}
        
        return result