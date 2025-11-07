"""
Модуль для трекинга людей с помощью ByteTrack.

Трекинг - это процесс присвоения уникальных ID людям и отслеживания их
между кадрами. Например, если человек с ID "p1" был на кадре 1 в позиции (100, 100),
и на кадре 2 в позиции (105, 102), трекер должен понять, что это тот же человек
и сохранить ему ID "p1", а не присвоить новый ID "p2".

Как это работает:
1. Детектор находит людей на каждом кадре (но не знает, кто есть кто)
2. Трекер сравнивает новые детекции со старыми треками
3. Если новый bounding box похож на старый (пересекается, близко расположен) -
   это тот же человек, обновляем его позицию
4. Если новый bounding box не похож ни на один старый - это новый человек,
   создаем новый ID (p1, p2, p3...)

ByteTrack - это современный алгоритм трекинга, который использует:
- IoU (Intersection over Union) - мера пересечения прямоугольников
- Kalman фильтр - для предсказания движения
- Иерархическое сопоставление - сначала высокоуверенные детекции, потом низкоуверенные

Если ByteTrack недоступен, используется упрощенный трекер на основе IoU.
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
try:
    # Пытаемся импортировать ByteTrack
    # ByteTrack может быть не установлен, поэтому используем try-except
    from bytetrack import BYTETracker
except ImportError:
    # Если ByteTrack не установлен, используем упрощенный трекер
    BYTETracker = None


class PersonTracker:
    """
    Класс для трекинга людей с уникальными ID.
    
    Трекер работает следующим образом:
    1. При первом появлении человека присваивает ему ID (p1, p2, p3...)
    2. Отслеживает человека между кадрами, сохраняя его ID
    3. Если человек исчезает на несколько кадров, но потом появляется -
       пытается восстановить его ID
    4. При создании нового ID проверяет базу данных на наличие недавно исчезнувших людей
       и использует их ID вместо создания новых
    """
    
    def __init__(self, track_thresh: float = 0.5, track_buffer: int = 30, match_thresh: float = 0.8,
                 database=None, room_name: str = None):
        """
        Инициализация трекера.
        
        Args:
            track_thresh: Порог для начала трекинга (минимальная уверенность детекции)
            track_buffer: Буфер для трекинга - сколько кадров хранить информацию
                         о человеке после его исчезновения
            match_thresh: Порог для сопоставления треков - минимальное пересечение
                         (IoU) для того, чтобы считать, что это тот же человек
            database: Экземпляр базы данных для восстановления старых ID (опционально)
            room_name: Имя комнаты для поиска недавно исчезнувших людей (опционально)
        """
        # Сохраняем ссылку на БД и имя комнаты для восстановления ID
        self.database = database
        self.room_name = room_name
        
        if BYTETracker is None:
            # ByteTrack не установлен - используем упрощенный трекер
            print("Предупреждение: ByteTrack не установлен. Используется упрощенный трекер.")
            self.tracker = None
            self.use_simple_tracker = True
            # Инициализируем next_id на основе существующих ID в БД
            self.next_id = self._initialize_next_id()
            # Словарь активных треков: {track_id: (x1, y1, x2, y2, confidence, last_seen_frame)}
            self.tracks = {}
            # Словарь "замороженных" треков - людей, которые исчезли, но могут вернуться
            # Формат: {track_id: (x1, y1, x2, y2, confidence, last_seen_frame, person_id)}
            self.frozen_tracks = {}
            self.frame_count = 0  # Счетчик кадров для определения старых треков
        else:
            # ByteTrack установлен - используем его
            self.tracker = BYTETracker(
                track_thresh=track_thresh,
                track_buffer=track_buffer,
                match_thresh=match_thresh
            )
            self.use_simple_tracker = False
            # Инициализируем next_id на основе существующих ID в БД
            self.next_id = self._initialize_next_id()
            # Маппинг внутренних ID ByteTrack на наши ID (p1, p2, ...)
            # ByteTrack использует свои внутренние ID, мы преобразуем их в наш формат
            self.id_mapping = {}
            self.frozen_tracks = {}
    
    def _initialize_next_id(self) -> int:
        """
        Инициализирует next_id на основе существующих ID в базе данных.
        
        Это предотвращает создание дубликатов ID при перезапуске системы.
        
        Returns:
            Следующий доступный ID (число)
        """
        if self.database is None:
            return 1
        
        try:
            # Получаем все существующие ID из БД
            existing_ids = self.database.get_all_person_ids()
            if not existing_ids:
                return 1
            
            # Извлекаем числа из ID (p1 -> 1, p2 -> 2, ...)
            numbers = []
            for person_id in existing_ids:
                if person_id.startswith('p'):
                    try:
                        num = int(person_id[1:])
                        numbers.append(num)
                    except ValueError:
                        pass
            
            if numbers:
                # Возвращаем максимальный ID + 1
                return max(numbers) + 1
            else:
                return 1
        except Exception as e:
            print(f"Ошибка при инициализации next_id: {e}")
            return 1
    
    def update(self, detections: List[Tuple[int, int, int, int, float]]) -> List[Tuple[int, int, int, int, str]]:
        """
        Обновить треки на основе новых детекций.
        
        Это главная функция трекера. Она вызывается для каждого кадра:
        1. Получает новые детекции (найденные люди) от детектора
        2. Сопоставляет их со старыми треками
        3. Обновляет существующие треки или создает новые
        4. Возвращает треки с ID
        
        Args:
            detections: Список детекций от детектора
                       Формат: [(x1, y1, x2, y2, confidence), ...]
            
        Returns:
            Список треков с ID
            Формат: [(x1, y1, x2, y2, person_id), ...]
            где person_id - строка типа "p1", "p2", "p3"...
        """
        # Если детекций нет (никого не нашли), просто очищаем старые треки
        if not detections:
            if self.use_simple_tracker:
                self.frame_count += 1
                # Перемещаем неактивные треки в замороженные (не удаляем)
                # УВЕЛИЧЕНО время хранения до 150 кадров
                tracks_to_remove = []
                for tid, track in self.tracks.items():
                    if self.frame_count - track[5] >= 150:
                        person_id = f"p{tid}"
                        self.frozen_tracks[tid] = (*track[:5], self.frame_count, person_id)
                        tracks_to_remove.append(tid)
                for tid in tracks_to_remove:
                    del self.tracks[tid]
                # Очищаем очень старые замороженные треки
                self.frozen_tracks = {tid: track for tid, track in self.frozen_tracks.items() 
                                     if self.frame_count - track[5] < 900}
            return []
        
        # Выбираем метод трекинга
        if self.use_simple_tracker:
            # Используем упрощенный трекер
            return self._update_simple(detections)
        else:
            # Используем ByteTrack
            return self._update_bytetrack(detections)
    
    def _update_simple(self, detections: List[Tuple[int, int, int, int, float]]) -> List[Tuple[int, int, int, int, str]]:
        """
        Упрощенный трекер для случаев, когда ByteTrack недоступен.
        
        Алгоритм работы:
        1. Для каждой новой детекции ищем ближайший старый трек по IoU
        2. Если IoU > порога (0.3) - это тот же человек, обновляем трек
        3. Если не нашли подходящий трек - создаем новый
        
        IoU (Intersection over Union) - это мера пересечения двух прямоугольников.
        IoU = 0.0 означает, что прямоугольники не пересекаются
        IoU = 1.0 означает, что прямоугольники полностью совпадают
        IoU = 0.5 означает, что они пересекаются на 50%
        """
        self.frame_count += 1  # Увеличиваем счетчик кадров
        tracks_output = []  # Результат - список треков с ID
        
        # Преобразуем детекции в удобный формат
        # Вычисляем центр и размер каждого bounding box
        current_detections = []
        for det in detections:
            x1, y1, x2, y2, conf = det
            center_x = (x1 + x2) / 2  # Центр по X
            center_y = (y1 + y2) / 2  # Центр по Y
            width = x2 - x1   # Ширина
            height = y2 - y1  # Высота
            current_detections.append((center_x, center_y, width, height, conf))
        
        # Сопоставляем новые детекции со старыми треками
        matched_tracks = set()  # Множество треков, которые уже сопоставлены
        
        # Для каждой новой детекции
        for det_idx, (cx, cy, w, h, conf) in enumerate(current_detections):
            best_match = None  # ID лучшего совпадения
            best_iou = 0.3     # Лучший IoU (порог минимального совпадения)
            
            # Ищем среди всех существующих треков
            for track_id, track in self.tracks.items():
                # Если трек уже сопоставлен, пропускаем его
                if track_id in matched_tracks:
                    continue
                
                # Получаем координаты старого трека
                tx1, ty1, tx2, ty2 = track[0], track[1], track[2], track[3]
                tcx = (tx1 + tx2) / 2  # Центр старого трека
                tcy = (ty1 + ty2) / 2
                
                # Вычисляем IoU (Intersection over Union) между новым и старым bounding box
                # Это показывает, насколько они пересекаются
                iou = self._calculate_iou(
                    (cx - w/2, cy - h/2, cx + w/2, cy + h/2),  # Новый bounding box
                    (tx1, ty1, tx2, ty2)  # Старый bounding box
                )
                
                # Если это лучшее совпадение (больше IoU), сохраняем
                if iou > best_iou:
                    best_iou = iou
                    best_match = track_id
            
            # Получаем координаты детекции
            x1, y1, x2, y2 = detections[det_idx][:4]
            
            if best_match is not None:
                # Нашли совпадение - обновляем существующий трек
                # Обновляем координаты, уверенность и номер кадра
                self.tracks[best_match] = (x1, y1, x2, y2, conf, self.frame_count)
                person_id = f"p{best_match}"  # Формируем ID (p1, p2, p3...)
                matched_tracks.add(best_match)  # Отмечаем, что трек сопоставлен
            else:
                # Не нашли совпадение в активных треках
                # Проверяем замороженные треки (недавно исчезнувшие)
                frozen_match = None
                frozen_best_iou = 0.3
                
                for frozen_tid, frozen_track in self.frozen_tracks.items():
                    tx1, ty1, tx2, ty2 = frozen_track[0], frozen_track[1], frozen_track[2], frozen_track[3]
                    iou = self._calculate_iou(
                        (cx - w/2, cy - h/2, cx + w/2, cy + h/2),
                        (tx1, ty1, tx2, ty2)
                    )
                    if iou > frozen_best_iou:
                        frozen_best_iou = iou
                        frozen_match = frozen_tid
                
                if frozen_match is not None:
                    # Нашли совпадение в замороженных треках - восстанавливаем ID
                    frozen_track = self.frozen_tracks[frozen_match]
                    track_id = frozen_match
                    person_id = frozen_track[6]  # Используем сохраненный person_id
                    # Перемещаем трек обратно в активные
                    self.tracks[track_id] = (x1, y1, x2, y2, conf, self.frame_count)
                    # Удаляем из замороженных
                    del self.frozen_tracks[frozen_match]
                    matched_tracks.add(track_id)
                    print(f"Восстановлен ID {person_id} из замороженных треков")
                else:
                    # Не нашли совпадение ни в активных, ни в замороженных треках
                    # Проверяем БД на наличие недавно исчезнувших людей
                    track_id = None
                    person_id = None
                    
                    if self.database is not None and self.room_name is not None:
                        # Ищем недавно исчезнувших людей из этой комнаты
                        recent_ids = self.database.find_recently_disappeared(self.room_name, max_seconds=60)
                        
                        # Пытаемся использовать один из недавно исчезнувших ID
                        # (в идеале нужна более умная логика сопоставления, но это базовый вариант)
                        if recent_ids:
                            # Берем первый доступный ID (можно улучшить логику сопоставления)
                            for recent_id in recent_ids:
                                # Проверяем, не используется ли этот ID в активных треках
                                recent_num = int(recent_id[1:]) if recent_id.startswith('p') else None
                                if recent_num is not None and recent_num not in self.tracks:
                                    track_id = recent_num
                                    person_id = recent_id
                                    print(f"Восстановлен ID {person_id} из базы данных")
                                    break
                    
                    # Если не нашли подходящий ID, создаем новый
                    if track_id is None:
                        track_id = self.next_id
                        self.next_id += 1
                        person_id = f"p{track_id}"
                    
                    # Создаем новый трек
                    self.tracks[track_id] = (x1, y1, x2, y2, conf, self.frame_count)
            
            # Добавляем трек в результат
            tracks_output.append((x1, y1, x2, y2, person_id))
        
        # Перемещаем неактивные треки в "замороженные" вместо удаления
        # УВЕЛИЧЕНО время хранения до 150 кадров (вместо 30) для лучшего запоминания
        # Это позволяет восстановить ID, если человек вернется в течение 5 секунд (при 30 FPS)
        tracks_to_remove = []
        for tid, track in self.tracks.items():
            if tid not in matched_tracks:
                # Трек не был сопоставлен - человек исчез
                if self.frame_count - track[5] >= 150:  # Увеличено с 30 до 150
                    # Исчез более 150 кадров назад - перемещаем в замороженные
                    # Сохраняем person_id для возможности восстановления
                    person_id = f"p{tid}"
                    # Храним: (x1, y1, x2, y2, conf, last_seen_frame, person_id)
                    self.frozen_tracks[tid] = (*track[:5], self.frame_count, person_id)
                    tracks_to_remove.append(tid)
        
        # Удаляем перемещенные треки из активных
        for tid in tracks_to_remove:
            del self.tracks[tid]
        
        # Очищаем очень старые замороженные треки (старше 900 кадров = ~30 секунд при 30 FPS)
        # Это позволяет восстановить ID даже после длительного отсутствия
        self.frozen_tracks = {tid: track for tid, track in self.frozen_tracks.items() 
                             if self.frame_count - track[5] < 900}  # Увеличено с 300 до 900
        
        return tracks_output
    
    def _update_bytetrack(self, detections: List[Tuple[int, int, int, int, float]]) -> List[Tuple[int, int, int, int, str]]:
        """
        Обновление треков с помощью ByteTrack.
        
        ByteTrack - это более сложный алгоритм трекинга, который:
        - Использует Kalman фильтр для предсказания движения
        - Иерархически сопоставляет детекции (сначала высокоуверенные, потом низкоуверенные)
        - Лучше справляется с перекрытиями и временными исчезновениями
        
        Этот метод вызывается только если ByteTrack установлен.
        """
        if not detections:
            return []
        
        # Преобразуем детекции в формат для ByteTrack
        # ByteTrack ожидает numpy array: [[x1, y1, x2, y2, score], ...]
        dets = np.array([[d[0], d[1], d[2], d[3], d[4]] for d in detections], dtype=np.float32)
        
        # Обновляем трекер ByteTrack
        # ByteTrack возвращает треки в формате: [[x1, y1, x2, y2, track_id], ...]
        tracks = self.tracker.update(dets)
        
        tracks_output = []
        for track in tracks:
            track_id = int(track[4])  # ID трека от ByteTrack (внутренний ID)
            x1, y1, x2, y2 = int(track[0]), int(track[1]), int(track[2]), int(track[3])
            
            # Маппим внутренний ID ByteTrack на наш формат (p1, p2, ...)
            # ByteTrack может переиспользовать ID (например, ID 5 может быть удален,
            # а потом присвоен новому человеку). Мы создаем постоянные ID.
            if track_id not in self.id_mapping:
                # Если это новый ID от ByteTrack, создаем новый наш ID
                self.id_mapping[track_id] = self.next_id
                self.next_id += 1
            
            # Формируем наш ID (p1, p2, p3...)
            person_id = f"p{self.id_mapping[track_id]}"
            tracks_output.append((x1, y1, x2, y2, person_id))
        
        return tracks_output
    
    def _calculate_iou(self, box1: Tuple[float, float, float, float], 
                       box2: Tuple[float, float, float, float]) -> float:
        """
        Вычисляет Intersection over Union (IoU) для двух bounding boxes.
        
        IoU - это мера пересечения двух прямоугольников.
        Формула: IoU = площадь_пересечения / площадь_объединения
        
        IoU используется для определения, насколько два bounding box'а похожи:
        - IoU = 0.0 - прямоугольники не пересекаются
        - IoU = 0.5 - пересекаются на 50%
        - IoU = 1.0 - полностью совпадают
        
        Args:
            box1: Первый bounding box (x1_min, y1_min, x1_max, y1_max)
            box2: Второй bounding box (x2_min, y2_min, x2_max, y2_max)
            
        Returns:
            IoU значение от 0.0 до 1.0
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # ============================================
        # ВЫЧИСЛЕНИЕ ПЕРЕСЕЧЕНИЯ
        # ============================================
        # Пересечение двух прямоугольников - это прямоугольник,
        # который находится внутри обоих прямоугольников
        
        # Левая граница пересечения - максимальная из левых границ
        inter_x_min = max(x1_min, x2_min)
        # Верхняя граница пересечения - максимальная из верхних границ
        inter_y_min = max(y1_min, y2_min)
        # Правая граница пересечения - минимальная из правых границ
        inter_x_max = min(x1_max, x2_max)
        # Нижняя граница пересечения - минимальная из нижних границ
        inter_y_max = min(y1_max, y2_max)
        
        # Если пересечения нет (границы неправильные), IoU = 0
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        # Вычисляем площадь пересечения
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # ============================================
        # ВЫЧИСЛЕНИЕ ОБЪЕДИНЕНИЯ
        # ============================================
        # Объединение - это общая площадь, покрываемая обоими прямоугольниками
        # Формула: площадь1 + площадь2 - площадь_пересечения
        # (Вычитаем пересечение, так как оно было посчитано дважды)
        
        # Площадь первого прямоугольника
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        # Площадь второго прямоугольника
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        # Площадь объединения
        union_area = box1_area + box2_area - inter_area
        
        # Защита от деления на ноль
        if union_area == 0:
            return 0.0
        
        # ============================================
        # ВЫЧИСЛЕНИЕ IoU
        # ============================================
        # IoU = площадь пересечения / площадь объединения
        return inter_area / union_area