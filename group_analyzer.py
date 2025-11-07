"""
Модуль для группового анализа людей.

Определяет группы людей, которые перемещаются вместе,
и отслеживает перемещения групп между комнатами.
"""
import time
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from database import Database


class GroupAnalyzer:
    """
    Класс для анализа групп людей.
    
    Определяет группы на основе:
    - Одновременного появления в одной комнате
    - Одновременного перемещения между комнатами
    - Близости по времени перемещений
    """
    
    def __init__(self, database: Database, group_window: float = 10.0):
        """
        Инициализация анализатора групп.
        
        Args:
            database: Экземпляр базы данных
            group_window: Окно времени для определения группы (секунды)
                        Если люди переместились в течение этого времени - они группа
        """
        self.db = database
        self.group_window = group_window
        
        # Активные группы: {group_id: {"members": Set[str], "current_room": str, "last_update": float}}
        self.active_groups: Dict[str, Dict] = {}
        self.next_group_id = 1
        
        # История перемещений для анализа групп
        # {person_id: [(room, timestamp), ...]}
        self.movement_history: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        
        import threading
        self.lock = threading.Lock()
    
    def analyze_movement(self, person_id: str, from_room: str, to_room: str, timestamp: float = None):
        """
        Анализировать перемещение человека на предмет группового движения.
        
        Args:
            person_id: ID человека
            from_room: Откуда переместился
            to_room: Куда переместился
            timestamp: Время перемещения (если None - текущее время)
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Добавляем в историю
        self.movement_history[person_id].append((to_room, timestamp))
        
        # Очищаем старую историю (старше group_window секунд)
        cutoff_time = timestamp - self.group_window
        self.movement_history[person_id] = [
            (room, t) for room, t in self.movement_history[person_id]
            if t > cutoff_time
        ]
        
        # Ищем людей, которые переместились в ту же комнату в похожее время
        with self.lock:
            potential_group_members = []
            
            for other_person, history in self.movement_history.items():
                if other_person == person_id:
                    continue
                
                # Проверяем, переместился ли этот человек в ту же комнату недавно
                for room, move_time in history:
                    if room == to_room and abs(move_time - timestamp) <= self.group_window:
                        potential_group_members.append((other_person, move_time))
                        break
            
            # Если есть потенциальные члены группы (2+ человека)
            if len(potential_group_members) >= 1:
                # Создаем или обновляем группу
                group_members = {person_id}
                for other_person, _ in potential_group_members:
                    group_members.add(other_person)
                
                # Проверяем, есть ли уже группа с этими людьми
                existing_group = None
                for group_id, group_data in self.active_groups.items():
                    if group_data["members"] == group_members:
                        existing_group = group_id
                        break
                
                if existing_group:
                    # Обновляем существующую группу
                    self.active_groups[existing_group]["current_room"] = to_room
                    self.active_groups[existing_group]["last_update"] = timestamp
                else:
                    # Создаем новую группу
                    group_id = f"Group{self.next_group_id}"
                    self.next_group_id += 1
                    self.active_groups[group_id] = {
                        "members": group_members,
                        "current_room": to_room,
                        "last_update": timestamp,
                        "from_room": from_room
                    }
                    
                    # Сохраняем в БД
                    members_list = sorted(list(group_members))
                    self.db.add_group_movement(group_id, from_room, to_room, members_list)
                    print(f"Группа {group_id} создана: {members_list} переместилась из {from_room} в {to_room}")
    
    def get_active_groups(self) -> Dict[str, Dict]:
        """
        Получить активные группы.
        
        Returns:
            Словарь активных групп
        """
        with self.lock:
            # Очищаем старые группы (не обновлялись более group_window секунд)
            current_time = time.time()
            to_remove = []
            for group_id, group_data in self.active_groups.items():
                if current_time - group_data["last_update"] > self.group_window * 3:
                    to_remove.append(group_id)
            
            for group_id in to_remove:
                del self.active_groups[group_id]
            
            return self.active_groups.copy()
