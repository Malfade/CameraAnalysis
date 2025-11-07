"""
Главный Flask сервер для системы подсчета людей и трекинга.

Этот файл является центральным компонентом системы. Он:
1. Инициализирует все компоненты (камеры, детектор, трекеры, БД)
2. Запускает потоки обработки видео для каждой камеры
3. Предоставляет веб-интерфейс через Flask
4. Отдает видео стримы и данные через API
"""
import cv2  # opencv - для работы с изображением и обработкой кадров
import threading  # для запуска многопоточности: потоки для камер и фоновые задачи
import time  # для sleep и меток времени
import json  # для работы с json-данными
import os    # для работы с ОС, например, получение pid процесса
import math  # для математических операций (расчет расстояний, sqrt)
from datetime import datetime  # для форматирования дат/времени
from flask import Flask, render_template, Response, jsonify, send_from_directory, request  # основные классы Flask
from flask_socketio import SocketIO, emit  # Websocket для real-time взаимодействия
from camera_stream import CameraStream  # класс для захвата кадров с камеры в отдельном потоке
from detector import PersonDetector     # детектор людей (YOLO и др.)
from tracker import PersonTracker      # трекер людей по кадрам (ByteTrack и др.)
from room_manager import RoomManager   # менеджер комнат, ведёт учёт кто где
from database import Database          # работа с базой данных (sqlite/другое)
from screenshot_manager import ScreenshotManager  # менеджер для хранения скриншотов событий
from video_recorder import VideoRecorder          # менеджер для записи видео
from group_analyzer import GroupAnalyzer          # анализирует группы людей (идут вместе)
import config   # конфиги проекта (пути, параметры, список камер и прочее)


# --------------------------------------------
# Создаем Flask приложение
# Flask - это веб-фреймворк для Python, который позволяет создавать веб-сервер
app = Flask(__name__)
app.config['SECRET_KEY'] = 'camera_analis_secret_key_2024'

# Создаем SocketIO для организации WebSocket соединений с браузером
# Это позволяет реализовать real-time: обновление данных на фронте без перезагрузки страницы
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ============================================
# ГЛОБАЛЬНЫЕ ОБЪЕКТЫ - используются во всем приложении
# ============================================

# Объект базы данных — хранит информацию о комнатах, людях, перемещениях, посещениях
database = Database()

# Менеджер скриншотов событий (делает и сохраняет изображения при входах/выходах/движении)
screenshot_manager = ScreenshotManager(
    screenshots_dir=config.SCREENSHOT_CONFIG["screenshots_dir"],
    auto_enabled=config.SCREENSHOT_CONFIG["auto_screenshots"],
    jpeg_quality=config.SCREENSHOT_CONFIG["jpeg_quality"]
)

# Видеорекордер для записи видео с камер
video_recorder = VideoRecorder(
    recordings_dir=config.VIDEO_RECORDING_CONFIG["recordings_dir"],
    codec=config.VIDEO_RECORDING_CONFIG["codec"],
    fps=config.VIDEO_RECORDING_CONFIG["fps"]
)

# Менеджер комнат: отслеживает состояние всех комнат, генерирует события перемещений, синхронизирует состояния
# Также принимает скриншот-менеджер, чтобы делать снимки движения/перемещений
room_manager = RoomManager(
    database, 
    config.MOVEMENT_CONFIG["movement_window"],
    screenshot_manager=screenshot_manager
)

# Анализатор групп — определяет людей, идущих/перемещающихся вместе (группы)
group_analyzer = GroupAnalyzer(database, group_window=10.0)

# Детектор людей (YOLO v8 или другой) — используется единожды на процесс, так как модель тяжёлая
detector = PersonDetector(
    config.DETECTION_CONFIG["model"],
    config.DETECTION_CONFIG["confidence_threshold"]
)

# Словарь трекеров по комнатам (чтобы уникальные id людей не пересекались между комнатами)
# {room_name: PersonTracker}
trackers = {}

# Словарь потоков-камер: для каждой комнаты отдельный CameraStream (или RTSP-поток)
# {room_name: CameraStream}
camera_streams = {}

# Последние обработанные кадры с аннотациями по всем комнатам
# {room_name: {"frame": annotated_frame, "tracks": [...]}}
frame_data = {}

# Метрики производительности: FPS, история количества людей и др.
# {room_name: { "fps": float, "people_count_history": [...], ...}}
performance_metrics = {}

# Позиции людей на карте: {room_name: [{"id": "p1", "x": 100, "y": 200, "distance_m": 2.3, "timestamp": 1234567890}, ...]}
# Координаты рассчитываются на основе центра bbox и масштабируются на карту
people_positions = {}

# Траектории движения людей: {person_id: [{"x": 100, "y": 200, "timestamp": 1234567890}, ...]}
# Храним последние N точек для отображения пути
person_trajectories = {}

# Лок (mutex) для потокобезопасного доступа к frame_data (чтобы несколько потоков не писали одновременно)
frame_lock = threading.Lock()
metrics_lock = threading.Lock()
positions_lock = threading.Lock()


def initialize_cameras():
    """
    Инициализация всех камер из конфигурации.

    Эта функция:
    1. Читает конфиг камер из config.CAMERAS_CONFIG
    2. Создаёт CameraStream для каждой камеры, запускает захват видео
    3. Создаёт трекеры для каждой комнаты (отдельно)
    4. Добавляет комнаты в базу данных (если их там ещё нет)
    """
    global camera_streams, trackers
    
    # Проходим по всем камерам, описанным в конфиге
    for cam_config in config.CAMERAS_CONFIG:
        room_name = cam_config["room_name"]      # Человекочитаемое имя комнаты (например, "Room1")
        camera_index = cam_config["camera_index"]  # Индекс камеры (0/1/2...) или RTSP ссылка

        # Добавляем комнату в БД (для инициализации комнат и корректной истории)
        database.add_room(room_name, str(camera_index))

        # Создаем CameraStream и запускаем поток захвата кадров
        try:
            # CameraStream - специальный класс, который асинхронно читает кадры и хранит последний актуальный
            camera = CameraStream(camera_index, room_name)
            camera.start()  # Запуск фонового потока захвата кадров с камеры
            camera_streams[room_name] = camera
            print(f"Камера {room_name} успешно инициализирована")
        except Exception as e:
            # Ошибка при инициализации камеры (например, если она занята или не подключена)
            print(f"Ошибка при инициализации камеры {room_name}: {e}")
            continue  # Пропускаем, продолжаем инициализацию остальных камер

        # Создаём трекер для этой комнаты
        # Персональные трекеры гарантируют независимость id-шников (p1, p2, ...) внутри комнаты
        trackers[room_name] = PersonTracker(
            track_thresh=config.TRACKING_CONFIG["track_thresh"],
            track_buffer=config.TRACKING_CONFIG["track_buffer"],
            match_thresh=config.TRACKING_CONFIG["match_thresh"],
            database=database,  # передаем БД для восстановления истории треков
            room_name=room_name
        )


def process_camera(room_name: str):
    """
    Функция обработки видеопотока одной камеры (одна функция — одна комната/камера).
    Выполняется в отдельном потоке для каждой камеры.
    Ее задача: захват кадра -> детекция -> трекинг -> обновление инфы -> аннотации -> обновление web-стрима.

    Args:
        room_name: Имя комнаты, к которой привязана камера
    """
    # Получаем соответствующий объект камеры и трекера
    camera = camera_streams.get(room_name)
    tracker = trackers.get(room_name)
    
    # Если камера или трекер не инициализированы (ошибка старта камеры) — выходим
    if not camera or not tracker:
        return
    
    # Основной бесконечный цикл обработки (работает до завершения процесса)
    while True:
        # Берём последний доступный кадр с камеры
        frame = camera.read()
        if frame is None:
            # Камера временно недоступна — подождать и попробовать снова
            time.sleep(0.1)
            continue
        
        # ======= ЭТАП 1: ДЕТЕКЦИЯ ЛЮДЕЙ НА КАДРЕ =======
        # Используем YOLO для поиска bounding box'ов людей: получаем список кортежей (x1, y1, x2, y2, confidence)
        detections = detector.detect(frame)
        
        # ======= ЭТАП 2: ТРЕКИНГ (ПРИСВОЕНИЕ ID) =======
        # Трекер анализирует: соответствие новых детекций старым id-шникам (p1, p2, ...),
        # возвращает: [(x1, y1, x2, y2, person_id), ...]
        tracks = tracker.update(detections)
        
        # Извлекаем id-шники людей из списка треков (для обновления состояния комнат и т.д.)
        person_ids = [track[4] for track in tracks]
        
        # ======= ЭТАП 2.5: РАСЧЕТ КООРДИНАТ ДЛЯ КАРТЫ =======
        # Рассчитываем позиции людей на карте на основе центра bbox
        frame_height, frame_width = frame.shape[:2]
        current_positions = []
        current_time = time.time()
        
        # Загружаем конфигурацию карты для расчета координат
        try:
            with open('room_map_config.json', 'r', encoding='utf-8') as f:
                map_config = json.load(f)
        except FileNotFoundError:
            map_config = None
        
        # Для каждого трека рассчитываем координаты на карте
        for x1, y1, x2, y2, person_id in tracks:
            # Центр bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            # Находим комнату в конфигурации карты
            room_config = None
            if map_config:
                for room in map_config.get('rooms', []):
                    if room['name'] == room_name:
                        room_config = room
                        break
            
            # Рассчитываем координаты на карте
            map_x = None
            map_y = None
            distance_m = None
            
            if room_config:
                # Масштабирование координат на карту
                map_x = room_config['x'] + (cx / frame_width) * room_config['width']
                map_y = room_config['y'] + (cy / frame_height) * room_config['height']
                
                # Расчет расстояния в метрах (если указаны размеры комнаты)
                if 'width_meters' in room_config and 'height_meters' in room_config:
                    meters_x = (cx / frame_width) * room_config['width_meters']
                    meters_y = (cy / frame_height) * room_config['height_meters']
                    # Расстояние от левого верхнего угла комнаты
                    distance_m = math.sqrt(meters_x**2 + meters_y**2)
                else:
                    # Если размеры не указаны, используем пиксельное расстояние
                    distance_px = math.sqrt((map_x - room_config['x'])**2 + (map_y - room_config['y'])**2)
                    # Примерное преобразование в метры (1 метр ≈ 100 пикселей на карте)
                    distance_m = distance_px / 100.0
            
            if map_x is not None and map_y is not None:
                current_positions.append({
                    "id": person_id,
                    "x": round(map_x, 2),
                    "y": round(map_y, 2),
                    "distance_m": round(distance_m, 2) if distance_m else None,
                    "timestamp": current_time
                })
                
                # Сохраняем траекторию (последние 50 точек)
                if person_id not in person_trajectories:
                    person_trajectories[person_id] = []
                person_trajectories[person_id].append({
                    "x": round(map_x, 2),
                    "y": round(map_y, 2),
                    "timestamp": current_time,
                    "room": room_name
                })
                # Ограничиваем траекторию последними 50 точками
                if len(person_trajectories[person_id]) > 50:
                    person_trajectories[person_id] = person_trajectories[person_id][-50:]
        
        # Сохраняем позиции для этой комнаты
        with positions_lock:
            people_positions[room_name] = current_positions
        
        # ======= ЭТАП 3: ВИЗУАЛИЗАЦИЯ/АННОТАЦИИ =======
        # Копируем кадр для добавления прямоугольников, подписей и другой overlay-аннотации
        annotated_frame = frame.copy()
        
        # Для каждого человека находим bbox, id, размещаем прямоугольник и подпись
        for x1, y1, x2, y2, person_id in tracks:
            # Рисуем зелёный прямоугольник вокруг человека
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Готовим текстовую подпись (id)
            label = person_id
            
            # Вычисляем размер текста, чтобы аккуратно нарисовать фон под подписью
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # Рисуем фон под текст над прямоугольником (чтобы надпись была читаемой)
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                (0, 255, 0),  # такой же цвет, чтобы гармонично смотрелось
                -1
            )

            # Поверх рисуем сам текст (черный, чтобы был виден на фоне)
            cv2.putText(
                annotated_frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )
        
        # Дополнительная информация — имя комнаты и текущий count людей (для визуального контроля)
        cv2.putText(
            annotated_frame, f"Room: {room_name}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        cv2.putText(
            annotated_frame, f"People: {len(person_ids)}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        
        # ======= ЭТАП 4: ОБНОВЛЕНИЕ ИНФОРМАЦИИ О КОМНАТЕ =======
        # Передаём список новых id-шников в RoomManager для обнаружения перемещений
        # и фотофиксации, а также обновления активных посетителей
        events = room_manager.update_room(room_name, person_ids, frame=annotated_frame)
        
        # Для каждого event, если это событие "перемещение", вызываем анализатор групп для агрегации совместных передвижений
        for event in events:
            if event["type"] == "move":
                group_analyzer.analyze_movement(
                    event["person_id"],
                    event["from_room"],
                    event["to_room"],
                    event["timestamp"]
                )
        
        # Если были события (например, вошли/вышли/переместились) — отправляем через сервер WebSocket всем клиентам
        if events:
            room_status = room_manager.get_all_rooms_status()
            socketio.emit('room_update', room_status)
            socketio.emit('events', events)
        
        # Отправляем обновление позиций через WebSocket для карты
        with positions_lock:
            all_positions = {}
            for room, positions in people_positions.items():
                all_positions[room] = positions
            socketio.emit('positions_update', all_positions)
        
        # ======= ЭТАП 5: ЗАПИСЬ ВИДЕО =======
        # Если идет запись для этой комнаты - добавляем кадр
        if video_recorder.is_recording(room_name):
            video_recorder.add_frame(room_name, annotated_frame)
        
        # ======= ЭТАП 6: СОХРАНЕНИЕ КАДРА ДЛЯ ВЕБ-СТРИМА =======
        # Сохраняем последний обработанный кадр + треки для MJPEG-стриминга или последующего анализа
        # Используем frame_lock, чтобы не было race condition при обращении из разных потоков
        with frame_lock:
            frame_data[room_name] = {
                "frame": annotated_frame,  # Кадр с аннотациями
                "tracks": tracks           # Список треков на этом кадре
            }
        
        # ======= ЭТАП 6: МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ =======
        # Сохраняем историю количества людей по времени, считаем FPS и прочее для панелей мониторинга
        current_time = time.time()
        with metrics_lock:
            if room_name not in performance_metrics:
                # Инициализировать статистику, если по комнате нет ещё статуса
                performance_metrics[room_name] = {
                    "fps": 0.0,
                    "people_count_history": [],
                    "last_update": current_time,
                    "frame_count": 0,
                    "fps_start_time": current_time
                }
            
            metrics = performance_metrics[room_name]
            metrics["frame_count"] += 1
            metrics["people_count_history"].append((current_time, len(person_ids)))
            
            # Пересчёт FPS (раз в секунду)
            elapsed = current_time - metrics["fps_start_time"]
            if elapsed >= 1.0:
                metrics["fps"] = metrics["frame_count"] / elapsed
                metrics["frame_count"] = 0
                metrics["fps_start_time"] = current_time
            
            # Обрезаем историю, если записей слишком много (оставляем последние 300 = ~5 минут при 1 запись/сек)
            if len(metrics["people_count_history"]) > 300:
                metrics["people_count_history"] = metrics["people_count_history"][-300:]
        
        # Ограничение нагрузки: маленькая задержка, чтобы не сжечь CPU и делать примерно 30 FPS
        time.sleep(0.03)


def cleanup_old_disappeared():
    """
    Периодическая очистка старых записей об исчезнувших людях.

    Когда человек вышел из кадра, его id временно попадает в disappeared_people.
    Если он через короткий промежуток времени не появился в другой комнате — мы его "забываем", 
    иначе фиксируем перемещение.

    Функция работает в своём отдельном потоке, периодически вызывает очистку (раз в N секунд).
    """
    while True:
        # Ждём 5 секунд между запусками очистки (можно настроить)
        time.sleep(5)
        # Удаляем людей, которые числятся как исчезнувшие больше N секунд (обычно 10 сек)
        room_manager.cleanup_old_disappeared()


# ============================================
# FLASK МАРШРУТЫ - веб-интерфейс
# ============================================

@app.route('/')
def index():
    """
    Главная страница с dashboard.

    Показывает: 
     - текущее количество людей в каждой комнате,
     - id всех присутствующих,
     - историю перемещений (модульно в виде таблицы),
     - кнопки для перехода к другому функционалу.
    """
    # render_template находит файл index.html в папке templates/
    # и возвращает HTML страницу пользователю
    return render_template('index.html')


@app.route('/video')
def video():
    """
    Страница с видео стримами.

    Показывает изображения/видео потоки с каждой камеры вживую.
    В шаблон передаётся список комнат для генерации блоков под каждую камеру.
    """
    # Передаем список имен комнат в шаблон для отображения
    return render_template('video.html', rooms=list(camera_streams.keys()))


@app.route('/statistics')
def statistics():
    """
    Страница со статистикой.

    Графики/отчёты по количеству людей/посещений и другим сводным данным.
    """
    return render_template('statistics.html')


@app.route('/map')
def map_page():
    """
    Страница с картой помещений.

    На ней показывается расположение комнат, позиционирование людей и/или камер.
    """
    return render_template('map.html')


@app.route('/api/rooms')
def api_rooms():
    """
    API endpoint для получения текущего состояния всех комнат (REST).

    Возвращает (пример):
    {
        "Room1": {"count": 3, "persons": ["p1", "p2", "p4"]},
        "Room2": {"count": 1, "persons": ["p3"]}
    }

    Чаще всего вызывается фронтом в режиме polling или через WebSocket-события.
    """
    # Получаем актуальный статус всех комнат (число людей, кто внутри)
    status = room_manager.get_all_rooms_status()
    # jsonify преобразует dict в корректный для браузера JSON
    return jsonify(status)


@app.route('/api/movements')
def api_movements():
    """
    API для получения истории перемещений между комнатами (последние 100 записей).

    Возвращает список перемещений вида:
    [
        {
            "timestamp": "2024-01-15 11:32:10",
            "time": "11:32:10",
            "person_id": "p3",
            "from_room": "Room1",
            "to_room": "Room2"
        },
        ...
    ]

    Используется для построения таблиц перемещений на dashboard или для аналитики.
    """
    # Получаем из БД записи последнее 100 перемещений людей
    movements = database.get_movements(limit=100)
    
    # Для красоты — отрезаем дату, оставляем только время для вывода на фронт
    for movement in movements:
        timestamp = movement['timestamp']
        if isinstance(timestamp, str):
            # timestamp как строка "2024-01-15 11:32:10" -> "11:32:10"
            movement['time'] = timestamp.split(' ')[1] if ' ' in timestamp else timestamp
        else:
            movement['time'] = str(timestamp)
    
    return jsonify(movements)


@app.route('/api/active_visits')
def api_active_visits():
    """
    API для получения активных посещений (кто сейчас в какой комнате и сколько времени).

    Возвращает список визитов (для подсветки, "кто внутри").
    """
    visits = database.get_active_visits()
    # Форматируем время входа (отдельно поле только время), а также округляем длительность визита
    for visit in visits:
        enter_time = visit['enter_time']
        if isinstance(enter_time, str):
            visit['enter_time_str'] = enter_time.split(' ')[1] if ' ' in enter_time else enter_time
        else:
            visit['enter_time_str'] = str(enter_time)
        visit['duration_min'] = round(visit.get('duration_min', 0), 2)
    return jsonify(visits)


@app.route('/api/statistics/<room_name>')
def api_statistics(room_name):
    """
    API endpoint для получения почасовой/посуточной статистики по конкретной комнате.

    Args:
        room_name: Имя комнаты, для которой нужна статистика

    Query параметры:
        hours: За сколько часов собрать статистику (по-умолчанию 24)
    """
    from flask import request
    hours = int(request.args.get('hours', 24))
    stats = database.get_room_statistics(room_name, hours)
    return jsonify(stats)


@app.route('/api/export/movements')
def api_export_movements():
    """
    API endpoint для скачивания истории перемещений людей в формате CSV-файла.

    Query параметры:
        limit: Максимальное количество последних записей для выгрузки (по умолчанию 1000)
    """
    from flask import request, send_file
    import csv
    import io
    
    # Получаем лимит записей из параметров запроса (или 1000 по-умолчанию)
    limit = int(request.args.get('limit', 1000))
    movements = database.get_movements(limit=limit)
    
    # Создаём в памяти строковый (Unicode) CSV-файл
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=['timestamp', 'person_id', 'from_room', 'to_room'])
    writer.writeheader()
    
    # Записываем все движения из истории по полям
    for movement in movements:
        writer.writerow({
            'timestamp': movement['timestamp'],
            'person_id': movement['person_id'],
            'from_room': movement['from_room'] or '',
            'to_room': movement['to_room']
        })
    
    # Перематываем позицию в начало, далее перекладываем в BytesIO
    output.seek(0)
    mem = io.BytesIO()
    mem.write(output.getvalue().encode('utf-8'))
    mem.seek(0)

    # Генерируем уникальное имя файла с текущей датой-временем экспорта
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"movements_export_{timestamp}.csv"
    
    # Отправляем файл клиенту через Flask send_file (в виде attachment)
    return send_file(
        mem,
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    )


@app.route('/api/camera_status')
def api_camera_status():
    """
    API endpoint для получения статусов всех камер (работает/ошибка, статус соединения и др).

    Возвращает словарь: {room_name: status_dict}
    """
    status = {}
    for room_name, camera_stream in camera_streams.items():
        # get_status() возвращает детальную информацию о работе камеры
        status[room_name] = camera_stream.get_status()
    return jsonify(status)


@app.route('/api/room_map')
def api_room_map():
    """
    API для получения конфигурации карты помещений.
    
    Returns:
        JSON с конфигурацией карты из room_map_config.json
    """
    try:
        with open('room_map_config.json', 'r', encoding='utf-8') as f:
            map_config = json.load(f)
        return jsonify(map_config)
    except FileNotFoundError:
        return jsonify({"error": "room_map_config.json not found"}), 404
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Invalid JSON in room_map_config.json: {str(e)}"}), 500


@app.route('/api/performance_metrics')
def api_performance_metrics():
    """
    API endpoint для получения производственных метрик (сколько fps, история по количеству людей и т.д.)

    Можно кастомизировать:
      - hours: за сколько часов история (по умолчанию 1)
      - max_points: максимальное количество точек истории

    Возвращает:
      {room_name: {fps, people_count_history, current_count}}
    """
    from flask import request
    
    # Читаем параметры запроса: hours - история, max_points - сжатие истории
    hours = float(request.args.get('hours', 1.0))  # За какое количество последних часов брать данные
    max_points = int(request.args.get('max_points', 100))  # Ограничение/прореживание истории для фронта

    current_time = time.time()
    cutoff_time = current_time - (hours * 3600)
    
    result = {}
    with metrics_lock:
        for room_name, metrics in performance_metrics.items():
            # Фильтруем историю по cutoff_time (только свежие записи)
            filtered_history = [
                (ts, count) for ts, count in metrics["people_count_history"]
                if ts >= cutoff_time
            ]
            
            # Если точек слишком много - прореживаем по step
            if len(filtered_history) > max_points:
                step = len(filtered_history) // max_points
                filtered_history = filtered_history[::step]
            
            # Формируем отдаваемый результат для каждой комнаты
            result[room_name] = {
                "fps": round(metrics["fps"], 2),
                "people_count_history": [
                    {"time": ts, "count": count} for ts, count in filtered_history
                ],
                "current_count": filtered_history[-1][1] if filtered_history else 0
            }
    
    return jsonify(result)


@app.route('/api/system_load')
def api_system_load():
    """
    API endpoint для получения текущей загрузки CPU/RAM на сервере и памяти текущего процесса.

    Использует библиотеку psutil (если она есть).
    """
    try:
        import psutil
        import os
        
        # Получаем процент загрузки CPU за небольшой интервал (0.1 секунды)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Получаем объект с информацией по ОЗУ
        memory = psutil.virtual_memory()
        
        # Информация по текущему Python-процессу (сколько он занимает памяти)
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info().rss / 1024 / 1024  # в мегабайтах

        # Возвращаем json со сводной инфой
        return jsonify({
            "cpu_percent": round(cpu_percent, 2),
            "memory_percent": round(memory.percent, 2),
            "memory_total_gb": round(memory.total / 1024 / 1024 / 1024, 2),
            "memory_used_gb": round(memory.used / 1024 / 1024 / 1024, 2),
            "process_memory_mb": round(process_memory, 2)
        })
    except ImportError:
        # Если psutil не установлен — отдаём фиктивные данные и сообщение об ошибке
        return jsonify({
            "error": "psutil not installed",
            "cpu_percent": 0,
            "memory_percent": 0
        })


@app.route('/api/positions')
def api_positions():
    """
    API для получения позиций всех людей на карте.
    
    Returns:
        JSON с позициями по комнатам:
        {
            "Room1": [
                {"id": "p1", "x": 100, "y": 200, "distance_m": 2.3, "timestamp": 1234567890},
                ...
            ],
            ...
        }
    """
    with positions_lock:
        # Возвращаем копию, чтобы избежать проблем с конкурентным доступом
        return jsonify(people_positions.copy())


@app.route('/api/screenshot/<room_name>', methods=['POST'])
def api_screenshot(room_name):
    """
    API для создания ручного скриншота с камеры.
    
    Args:
        room_name: Имя комнаты
        
    Returns:
        JSON с путем к сохраненному файлу
    """
    try:
        with frame_lock:
            if room_name not in frame_data:
                return jsonify({"error": f"Нет данных для комнаты {room_name}"}), 404
            
            frame = frame_data[room_name]["frame"]
            if frame is None:
                return jsonify({"error": "Кадр недоступен"}), 404
        
        # Сохраняем скриншот
        filepath = screenshot_manager.save_manual_screenshot(frame, room_name, "manual")
        
        if filepath:
            filename = os.path.basename(filepath)
            return jsonify({
                "success": True,
                "filepath": filepath,
                "filename": filename,
                "url": f"/screenshots/{filename}"
            })
        else:
            return jsonify({"error": "Не удалось сохранить скриншот"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/recording/<room_name>/start', methods=['POST'])
def api_start_recording(room_name):
    """
    API для начала записи видео с камеры.
    
    Args:
        room_name: Имя комнаты
        
    Returns:
        JSON с путем к файлу записи
    """
    try:
        # Получаем размер кадра из последнего кадра
        with frame_lock:
            if room_name not in frame_data:
                return jsonify({"error": f"Нет данных для комнаты {room_name}"}), 404
            
            frame = frame_data[room_name]["frame"]
            if frame is None:
                return jsonify({"error": "Кадр недоступен"}), 404
            
            frame_height, frame_width = frame.shape[:2]
        
        # Начинаем запись
        filepath = video_recorder.start_recording(room_name, frame_width, frame_height)
        
        if filepath:
            filename = os.path.basename(filepath)
            return jsonify({
                "success": True,
                "filepath": filepath,
                "filename": filename,
                "message": f"Запись начата для {room_name}"
            })
        else:
            return jsonify({"error": "Не удалось начать запись"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/recording/<room_name>/stop', methods=['POST'])
def api_stop_recording(room_name):
    """
    API для остановки записи видео.
    
    Args:
        room_name: Имя комнаты
        
    Returns:
        JSON с путем к сохраненному файлу
    """
    try:
        filepath = video_recorder.stop_recording(room_name)
        
        if filepath:
            filename = os.path.basename(filepath)
            return jsonify({
                "success": True,
                "filepath": filepath,
                "filename": filename,
                "url": f"/recordings/{filename}",
                "message": f"Запись остановлена для {room_name}"
            })
        else:
            return jsonify({"error": f"Запись не активна для {room_name}"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/recording/<room_name>/status', methods=['GET'])
def api_recording_status(room_name):
    """
    API для получения статуса записи.
    
    Args:
        room_name: Имя комнаты
        
    Returns:
        JSON со статусом записи
    """
    try:
        is_recording = video_recorder.is_recording(room_name)
        info = video_recorder.get_recording_info(room_name) if is_recording else None
        
        # Преобразуем datetime в строку для JSON сериализации
        if info and 'start_time' in info:
            if hasattr(info['start_time'], 'strftime'):
                info['start_time'] = info['start_time'].strftime("%Y-%m-%d %H:%M:%S")
        
        return jsonify({
            "is_recording": is_recording,
            "info": info
        })
    except Exception as e:
        import traceback
        print(f"Ошибка в api_recording_status: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/recordings/<filename>')
def serve_recording(filename):
    """Сервинг записей видео."""
    return send_from_directory(video_recorder.recordings_dir, filename)


@app.route('/api/trajectories')
def api_trajectories():
    """
    API для получения траекторий движения людей.
    
    Query параметры:
        person_id: ID конкретного человека (опционально)
    
    Returns:
        JSON с траекториями:
        {
            "p1": [{"x": 100, "y": 200, "timestamp": 1234567890, "room": "Room1"}, ...],
            ...
        }
    """
    person_id = request.args.get('person_id')
    
    if person_id:
        # Возвращаем траекторию конкретного человека
        trajectory = person_trajectories.get(person_id, [])
        return jsonify({person_id: trajectory})
    else:
        # Возвращаем все траектории
        return jsonify(person_trajectories.copy())


@app.route('/video_feed/<room_name>')
def video_feed(room_name):
    """
    MJPEG stream endpoint для live-видео по каждой камере/комнате.

    Это HTTP endpoint, возвращающий поток изображений (multipart/x-mixed-replace).
    Браузер сам обновляет картинку и воспроизводит поток в реальном времени.

    Args:
        room_name: Комната (имя), для которой вернуть поток MJPEG
    """
    def generate():
        """
        Генератор MJPEG потока.

        В бесконечном цикле — забирает последний кадр, кодирует и отправляет наружу.
        """
        while True:
            # Захватываем мультипоточный доступ к кадрам
            with frame_lock:
                if room_name in frame_data:
                    # Если есть актуальный кадр для комнаты — берём его (обработанный с аннотациями)
                    frame = frame_data[room_name]["frame"]
                else:
                    # Иначе создаём чёрный кадр с надписью "Camera not available"
                    frame = cv2.zeros((480, 640, 3), dtype=cv2.uint8)
                    cv2.putText(
                        frame, f"Camera {room_name} not available", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
                    )
            # Кодируем картинку в JPEG (масив байт)
            # Флажок [cv2.IMWRITE_JPEG_QUALITY, 85] настраивает качество
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            if ret:
                # Готовим блок MJPEG-ответа для браузера: \r\n — обязательный разделитель
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                # Если не получилось закодировать — ждем и повторяем
                time.sleep(0.1)
    
    # Возвращаем потоковый HTTP-ответ со special mime-типом для MJPEG
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ============================================
# ТОЧКА ВХОДА - запуск приложения
# ============================================
if __name__ == '__main__':
    # Этот блок выполняется только при прямом запуске скрипта (не при импорте в качестве модуля)
    
    print("Инициализация системы...")
    
    # Шаг 1: Инициализация всех камер (разворачиваем объекты, стартуем потоки захвата кадров)
    initialize_cameras()
    
    # Если ни одной камеры не удалось инициализировать — выдаём ошибку и завершаемся
    if not camera_streams:
        print("Ошибка: Не удалось инициализировать ни одной камеры!")
        print("Проверьте, что камера подключена и не используется другим приложением.")
        exit(1)
    
    # Шаг 2: Запускаем отдельный поток обработки для каждой камеры (worker-контур именно для видеоаналитики)
    for room_name in camera_streams.keys():
        # threading.Thread позволяет создать отдельный поток внутри процесса
        # target — запускаемая функция, args=(room_name,) — передаём имя комнаты
        # daemon=True — если главный процесс завершится, потоки завершаются автоматически
        thread = threading.Thread(target=process_camera, args=(room_name,), daemon=True)
        thread.start()
        print(f"Запущен поток обработки для {room_name}")
    
    # Шаг 3: Фоновый поток для очистки исчезнувших ID-шников
    # Это нужно для корректного закрытия визитов и освобождения памяти
    cleanup_thread = threading.Thread(target=cleanup_old_disappeared, daemon=True)
    cleanup_thread.start()
    
    print("Запуск Flask сервера...")
    print(f"Откройте браузер: http://localhost:{config.SERVER_CONFIG['port']}")
    print("Нажмите Ctrl+C для остановки")
    
    # Шаг 4: Стартуем web-сервер с поддержкой WebSocket real-time уведомлений
    # socketio.run запускает Flask-сервер + WebSocket (для событий, обновлений данных и т.д.)
    socketio.run(
        app,
        host=config.SERVER_CONFIG['host'],   # например, '0.0.0.0' — слушать все интерфейсы
        port=config.SERVER_CONFIG['port'],   # например, 5000
        debug=config.SERVER_CONFIG['debug'], # True — режим отладки, False — боевой режим
        allow_unsafe_werkzeug=True           # разрешить запуск под основным Werkzeug (для dev)
    )