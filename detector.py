"""
Модуль для детекции людей с помощью YOLOv8.

YOLO (You Only Look Once) - это нейронная сеть для детекции объектов.
YOLOv8 - это последняя версия, которая очень быстро и точно находит объекты на изображении.

Как это работает:
1. YOLO была обучена на миллионах изображений и знает, как выглядит человек
2. Когда мы передаем кадр, YOLO сканирует его и находит всех людей
3. Для каждого человека возвращает координаты прямоугольника (bounding box)
   и уверенность (confidence) - насколько она уверена, что это человек

Этот модуль оборачивает YOLO в удобный класс для работы в нашей системе.
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional
import torch


class PersonDetector:
    """
    Класс для детекции людей с помощью YOLOv8.
    
    Этот класс:
    1. Загружает предобученную модель YOLO
    2. Использует её для поиска людей на кадрах
    3. Возвращает координаты найденных людей
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Инициализация детектора.
        
        При создании объекта:
        1. Загружается модель YOLO (файл .pt)
        2. Если файла нет, он автоматически скачивается
        3. Настраивается порог уверенности
        
        Args:
            model_path: Путь к модели YOLO или имя предобученной модели
                       - "yolov8n.pt" - nano (быстрая, менее точная)
                       - "yolov8s.pt" - small (средняя)
                       - "yolov8m.pt" - medium (медленнее, точнее)
                       - "yolov8l.pt" - large (еще точнее)
                       - "yolov8x.pt" - xlarge (самая точная, самая медленная)
            confidence_threshold: Порог уверенности (0.0-1.0)
                                 0.5 означает "вернуть только те детекции,
                                 где уверенность >= 50%"
        """
        # Загружаем модель YOLO
        # При первом запуске файл автоматически скачается
        # Модель - это файл с весами нейронной сети (обученные параметры)
        self.model = YOLO(model_path)
        
        # Порог уверенности - минимальная уверенность для детекции
        # Если YOLO уверена на 30%, что это человек, а порог 50% - детекция отбрасывается
        self.confidence_threshold = confidence_threshold
        
        # В COCO dataset (на котором обучалась YOLO) класс "person" имеет ID 0
        # COCO - это большой набор данных с 80 классами объектов
        # Класс 0 = person, класс 1 = bicycle, класс 2 = car, и т.д.
        self.person_class_id = 0
        
        # Проверяем, доступна ли GPU (видеокарта)
        # GPU может ускорить обработку в 10-100 раз
        # torch.cuda.is_available() возвращает True, если есть CUDA-совместимая видеокарта
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Детектор использует: {self.device}")
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """
        Вычисляет Intersection over Union (IoU) между двумя bounding boxes.
        
        IoU используется для определения, насколько два прямоугольника перекрываются.
        Если один bbox полностью находится внутри другого - это, вероятно, часть тела (рука, нога).
        
        Args:
            box1: (x1, y1, x2, y2) первого bbox
            box2: (x1, y1, x2, y2) второго bbox
        
        Returns:
            Значение IoU от 0.0 до 1.0 (1.0 = полное перекрытие)
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Вычисляем площадь пересечения
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Вычисляем площади каждого bbox
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Объединение = площадь1 + площадь2 - пересечение
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _is_box_inside(self, inner_box: Tuple[int, int, int, int], outer_box: Tuple[int, int, int, int]) -> bool:
        """
        Проверяет, находится ли один bbox полностью внутри другого.
        
        Это помогает отфильтровать детекции частей тела (руки, ноги),
        которые YOLO может детектировать как отдельных людей.
        
        Args:
            inner_box: Внутренний bbox (проверяемый)
            outer_box: Внешний bbox (контейнер)
        
        Returns:
            True если inner_box полностью внутри outer_box
        """
        x1_i, y1_i, x2_i, y2_i = inner_box
        x1_o, y1_o, x2_o, y2_o = outer_box
        
        return (x1_i >= x1_o and y1_i >= y1_o and x2_i <= x2_o and y2_i <= y2_o)
    
    def _filter_detections(self, detections: List[Tuple[int, int, int, int, float]]) -> List[Tuple[int, int, int, int, float]]:
        """
        Фильтрует детекции для удаления ложных срабатываний (части тела, маленькие объекты).
        
        Фильтрация включает:
        1. Удаление слишком маленьких bbox (руки, ноги)
        2. Удаление bbox, которые находятся внутри других (части тела внутри тела)
        3. Удаление bbox с неправильным соотношением сторон (люди обычно вертикальные)
        4. Удаление bbox, которые сильно перекрываются с другими (дубликаты)
        
        Args:
            detections: Список детекций [(x1, y1, x2, y2, confidence), ...]
        
        Returns:
            Отфильтрованный список детекций
        """
        if not detections:
            return detections
        
        filtered = []
        
        # Сортируем по размеру (от больших к маленьким) и по уверенности (от высокой к низкой)
        # Это важно, чтобы сначала обрабатывать большие bbox (тела), а потом маленькие (части)
        detections_sorted = sorted(
            detections, 
            key=lambda d: ((d[2] - d[0]) * (d[3] - d[1]) * d[4]),  # площадь * уверенность
            reverse=True
        )
        
        for i, det in enumerate(detections_sorted):
            x1, y1, x2, y2, conf = det
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            # Фильтр 1: Минимальный размер bbox (абсолютные значения)
            # Руки обычно меньше 150x150 пикселей
            if area < 150 * 150:  # Минимум 22500 пикселей
                continue
            
            # Фильтр 2: Минимальная высота
            # Люди обычно имеют минимальную высоту (например, 150 пикселей)
            if height < 150:
                continue
            
            # Фильтр 3: Соотношение сторон
            # Люди обычно вертикальные (высота > ширины)
            aspect_ratio = width / height if height > 0 else 0
            # Если ширина больше высоты более чем в 1.2 раза - это, вероятно, не человек
            if aspect_ratio > 1.2:  # Ужесточено с 1.5 до 1.2
                continue
            # Если высота меньше ширины более чем в 3 раза - тоже подозрительно
            if aspect_ratio < 0.33:  # Слишком узкий
                continue
            
            # Фильтр 4: Проверка на вложенность и перекрытие
            # Если этот bbox находится внутри другого или сильно перекрывается - пропускаем
            is_duplicate = False
            for other_det in filtered:
                other_box = (other_det[0], other_det[1], other_det[2], other_det[3])
                current_box = (x1, y1, x2, y2)
                other_area = (other_det[2] - other_det[0]) * (other_det[3] - other_det[1])
                
                # Проверка 1: Полная вложенность
                if self._is_box_inside(current_box, other_box):
                    # Если текущий bbox полностью внутри другого и меньше - это часть тела
                    if area < other_area * 0.8:  # Ужесточено: текущий должен быть меньше на 20%
                        is_duplicate = True
                        break
                
                # Проверка 2: Высокое перекрытие (IoU)
                iou = self._calculate_iou(current_box, other_box)
                if iou > 0.5:  # Если перекрытие больше 50%
                    # Если текущий bbox меньше и перекрывается - это дубликат или часть тела
                    if area < other_area * 0.7:  # Текущий должен быть значительно меньше
                        is_duplicate = True
                        break
                    # Если они примерно одинакового размера, но сильно перекрываются - оставляем более уверенный
                    elif abs(area - other_area) < other_area * 0.3:  # Разница менее 30%
                        if conf < other_det[4]:  # Текущий менее уверенный
                            is_duplicate = True
                            break
            
            if is_duplicate:
                continue
            
            # Фильтр 5: Проверка на слишком маленькую площадь относительно других детекций
            # Если есть другие детекции, и текущая намного меньше - пропускаем
            if filtered:
                max_area = max((d[2] - d[0]) * (d[3] - d[1]) for d in filtered)
                if area < max_area * 0.3:  # Текущая детекция меньше 30% от максимальной
                    continue
            
            # Все проверки пройдены - добавляем детекцию
            filtered.append(det)
        
        return filtered
    
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Детекция людей на кадре.
        
        Это главная функция класса. Она принимает кадр (изображение) и возвращает
        список найденных людей с координатами.
        
        Процесс работы:
        1. Передаем кадр в YOLO модель
        2. YOLO анализирует кадр и находит всех людей
        3. Фильтруем результаты по порогу уверенности
        4. Возвращаем координаты bounding boxes
        
        Args:
            frame: Кадр изображения в формате numpy array
                   Формат BGR (Blue, Green, Red) - стандарт OpenCV
                   Размер: (высота, ширина, 3 канала цвета)
            
        Returns:
            Список кортежей, каждый кортеж содержит:
            - x1, y1: координаты левого верхнего угла прямоугольника
            - x2, y2: координаты правого нижнего угла прямоугольника
            - confidence: уверенность детекции (0.0-1.0)
            
            Пример: [(100, 50, 200, 300, 0.95), (350, 100, 450, 350, 0.87)]
            Это означает, что найдено 2 человека:
            - Первый: прямоугольник от (100, 50) до (200, 300), уверенность 95%
            - Второй: прямоугольник от (350, 100) до (450, 350), уверенность 87%
        """
        # Вызываем YOLO для детекции
        # model(frame, ...) - передаем кадр в модель
        # conf=... - порог уверенности (фильтруем детекции с низкой уверенностью)
        # classes=[self.person_class_id] - ищем только класс "person" (ID=0)
        # verbose=False - не выводить детальную информацию в консоль
        results = self.model(frame, conf=self.confidence_threshold, classes=[self.person_class_id], verbose=False)
        
        # Список для хранения детекций
        detections = []
        
        # Проверяем, есть ли результаты
        if results and len(results) > 0:
            # results[0] - первый (и единственный) результат
            # results[0].boxes - список всех найденных bounding boxes
            boxes = results[0].boxes
            
            # Проходим по всем найденным людям
            for box in boxes:
                # Получаем координаты bounding box
                # box.xyxy[0] - это tensor (массив) с координатами [x1, y1, x2, y2]
                # .cpu() - переносим с GPU на CPU (если использовалась GPU)
                # .numpy() - преобразуем tensor в обычный numpy array
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Получаем уверенность детекции
                # box.conf[0] - уверенность (confidence) в формате tensor
                confidence = box.conf[0].cpu().numpy()
                
                # Добавляем детекцию в список
                # Преобразуем координаты в целые числа (int) для удобства
                # confidence оставляем как float (0.0-1.0)
                detections.append((
                    int(x1), int(y1), int(x2), int(y2), float(confidence)
                ))
        
        # Возвращаем список всех найденных людей
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Tuple[int, int, int, int, float]]) -> np.ndarray:
        """
        Рисует bounding boxes на кадре.
        
        Эта функция используется для отладки - она рисует прямоугольники
        вокруг найденных людей прямо на изображении.
        
        В основной программе мы не используем эту функцию, так как рисуем
        аннотации в app.py. Но она может быть полезна для тестирования.
        
        Args:
            frame: Кадр изображения (не изменяется)
            detections: Список детекций из функции detect()
            
        Returns:
            Копия кадра с нарисованными bounding boxes и подписями
        """
        # Создаем копию кадра, чтобы не изменять оригинал
        frame_copy = frame.copy()
        
        # Рисуем каждый bounding box
        for x1, y1, x2, y2, conf in detections:
            # Рисуем зеленый прямоугольник вокруг человека
            # cv2.rectangle(изображение, (x1, y1), (x2, y2), цвет, толщина)
            # (0, 255, 0) - зеленый цвет в формате BGR
            # 2 - толщина линии в пикселях
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Создаем текст с уверенностью детекции
            # f"Person {conf:.2f}" - форматируем число с 2 знаками после запятой
            # Например: "Person 0.95"
            label = f"Person {conf:.2f}"
            
            # Рисуем текст над прямоугольником
            # cv2.putText(изображение, текст, позиция, шрифт, размер, цвет, толщина)
            cv2.putText(frame_copy, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Возвращаем кадр с рисунками
        return frame_copy