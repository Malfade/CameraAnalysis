"""
Модуль для работы с базой данных SQLite.

SQLite - это легковесная база данных, которая хранит данные в файле.
Не требует отдельного сервера, идеально подходит для небольших приложений.

База данных хранит три таблицы:
1. rooms - информация о комнатах (название, камера)
2. persons - текущее местоположение людей (ID, комната, время последнего обнаружения)
3. movements - история перемещений между комнатами

Структура данных:
- rooms: id, name, camera_index, created_at
- persons: person_id (PRIMARY KEY), current_room, last_seen
- movements: id, timestamp, person_id, from_room, to_room

Этот модуль предоставляет удобный интерфейс для работы с БД,
скрывая детали SQL запросов.
"""
import sqlite3              # Импортируем библиотеку для работы с SQLite
import threading            # Импортируем threading для обеспечения потокобезопасности (lock)
import json                 # Импортируем для работы с JSON-строкой (для хранения групповых перемещений)
from datetime import datetime, timedelta    # Для работы с датой и временем
from typing import List, Dict, Optional, Tuple   # Для аннотаций типов

class Database:
    """
    Класс для работы с базой данных SQLite.
    
    Этот класс:
    1. Создает и инициализирует базу данных
    2. Предоставляет методы для добавления/обновления/получения данных
    3. Обеспечивает потокобезопасность (lock для одновременного доступа)
    """
    
    def __init__(self, db_path: str = "camera_analis.db"):
        """
        Инициализация подключения к БД.
        
        При создании объекта:
        1. Сохраняется путь к файлу БД
        2. Создается lock для потокобезопасности
        3. Инициализируется схема БД (создаются таблицы, если их нет)
        
        Args:
            db_path: Путь к файлу базы данных
                    По умолчанию "camera_analis.db" в текущей директории
        """
        self.db_path = db_path
        # Lock для потокобезопасности (при доступе к БД из нескольких потоков)
        self.lock = threading.Lock()
        # Инициализируем схему БД (создаем все необходимые таблицы при запуске)
        self.init_database()
    
    def get_connection(self):
        """
        Получить соединение с БД.
        
        SQLite требует создать соединение (connection) перед каждым запросом.
        Эта функция создает новое соединение.
        
        Returns:
            Объект соединения с БД
        """
        # Создаём соединение с БД. Если файл не существует — он будет создан автоматически.
        conn = sqlite3.connect(self.db_path)
        # row_factory = sqlite3.Row позволяет обращаться к колонкам по имени (row['name'])
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        """
        Инициализация схемы базы данных.
        
        Создает таблицы, если их еще нет. Эта функция вызывается один раз
        при создании объекта Database.
        
        Таблицы:
        1. rooms - информация о комнатах
        2. persons - текущее местоположение людей
        3. movements - история перемещений
        """
        conn = self.get_connection()
        cursor = conn.cursor()  # Создаём объект cursor для выполнения SQL-запросов
        
        # ============================================
        # ТАБЛИЦА 1: ROOMS (КОМНАТЫ)
        # ============================================
        # Таблица для хранения данных о комнатах
        # id         - уникальный идентификатор комнаты (autoincrement)
        # name       - уникальное имя комнаты (например, "Room1"), обязательное поле
        # camera_index - индекс или url камеры, связанной с комнатой
        # created_at - дата и время создания комнаты
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rooms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,  -- Автоматически увеличивающийся ID
                name TEXT NOT NULL UNIQUE,             -- Имя комнаты (уникальное, обязательное)
                camera_index TEXT NOT NULL,            -- Индекс или URL камеры
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- Время создания записи
            )
        """)
        
        # ============================================
        # ТАБЛИЦА 2: PERSONS (ЛЮДИ)
        # ============================================
        # Таблица для хранения текущего положения людей
        # person_id    - уникальный идентификатор (например, "p1"), первичный ключ
        # current_room - текущая комната, где находится человек (может быть NULL, если не в комнате)
        # last_seen    - дата и время последнего обнаружения
        # Внешний ключ current_room указывает на поле name таблицы rooms
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                person_id TEXT PRIMARY KEY,            -- ID человека (p1, p2, p3...) - первичный ключ
                current_room TEXT,                     -- Текущая комната (может быть NULL)
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- Время последнего обнаружения
                FOREIGN KEY (current_room) REFERENCES rooms(name)  -- Связь с таблицей rooms
            )
        """)
        
        # ============================================
        # ТАБЛИЦА 3: MOVEMENTS (ПЕРЕМЕЩЕНИЯ)
        # ============================================
        # Таблица для хранения истории перемещений людей между комнатами
        # id        - уникальный идентификатор перемещения (autoincrement)
        # timestamp - когда произошло перемещение
        # person_id - идентификатор человека
        # from_room - из какой комнаты (NULL, если первое появление)
        # to_room   - в какую комнату переместился (обязательное поле)
        # Внешние ключи - на person_id из persons, from_room и to_room на name из rooms
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS movements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,  -- Автоматически увеличивающийся ID
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- Время перемещения
                person_id TEXT NOT NULL,               -- ID человека
                from_room TEXT,                        -- Откуда (может быть NULL для первого появления)
                to_room TEXT NOT NULL,                 -- Куда (обязательное поле)
                FOREIGN KEY (person_id) REFERENCES persons(person_id),  -- Связь с persons
                FOREIGN KEY (from_room) REFERENCES rooms(name),         -- Связь с rooms
                FOREIGN KEY (to_room) REFERENCES rooms(name)            -- Связь с rooms
            )
        """)
        
        # ============================================
        # ТАБЛИЦА 4: ROOM_VISITS (ВРЕМЯ ПРЕБЫВАНИЯ)
        # ============================================
        # Таблица для хранения времени пребывания людей в комнатах
        # id           - уникальный идентификатор посещения (autoincrement)
        # person_id    - идентификатор человека
        # room_name    - имя комнаты
        # enter_time   - время входа в комнату
        # exit_time    - время выхода из комнаты (NULL, если человек всё ещё в комнате)
        # duration_min - длительность пребывания в минутах (вычисляется при выходе)
        # Внешние ключи - person_id на persons, room_name на rooms
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS room_visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                room_name TEXT NOT NULL,
                enter_time TIMESTAMP NOT NULL,          -- Время входа
                exit_time TIMESTAMP,                    -- Время выхода (NULL если еще в комнате)
                duration_min REAL,                      -- Длительность в минутах
                FOREIGN KEY (person_id) REFERENCES persons(person_id),
                FOREIGN KEY (room_name) REFERENCES rooms(name)
            )
        """)
        
        # Добавляем индекс для быстрого поиска активных (текущих) посещений по person_id и exit_time.
        # Индекс ускоряет запросы, связанные с определением — находится ли человек сейчас в какой-либо комнате.
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_room_visits_active 
            ON room_visits(person_id, exit_time) 
            WHERE exit_time IS NULL
        """)
        
        # ============================================
        # ТАБЛИЦА 5: GROUP_MOVEMENTS (ГРУППОВЫЕ ПЕРЕМЕЩЕНИЯ)
        # ============================================
        # Таблица для хранения информации о перемещениях групп людей, а не отдельных персон
        # id        - уникальный идентификатор записи (autoincrement)
        # group_id  - идентификатор группы (логический, например "Group1")
        # from_room - из какой комнаты переместилась группа
        # to_room   - в какую комнату переместилась группа
        # members   - JSON-строка со списком ID людей, входящих в группу
        # timestamp - время перемещения
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS group_movements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id TEXT NOT NULL,
                from_room TEXT,
                to_room TEXT NOT NULL,
                members TEXT NOT NULL,  -- JSON список ID людей
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (to_room) REFERENCES rooms(name)
            )
        """)
        
        # commit() сохраняет изменения в БД.
        # Без commit() изменения не будут записаны на диск и потеряются при закрытии соединения.
        conn.commit()
        conn.close()  # Закрываем соединение после завершения инициализации таблиц
    
    def add_room(self, name: str, camera_index: str) -> bool:
        """
        Добавить комнату в базу данных.
        
        Если комната с таким именем уже существует, она не будет добавлена
        (благодаря UNIQUE constraint).
        
        Args:
            name: Имя комнаты (например, "Room1")
            camera_index: Индекс или URL камеры (например, "0" или "rtsp://...")
            
        Returns:
            True если комната успешно добавлена, False если уже существует
        """
        with self.lock:  # Используем lock для потокобезопасности, чтобы несколько потоков не добавили одну и ту же комнату одновременно
            conn = self.get_connection()
            cursor = conn.cursor()
            try:
                # Добавляем новую комнату — если уже есть комната с таким именем, будет ошибка (UNIQUE)
                cursor.execute(
                    "INSERT INTO rooms (name, camera_index) VALUES (?, ?)",
                    (name, camera_index)  # Передаём параметры через плейсхолдеры для защиты от SQL-инъекций
                )
                conn.commit()  # Сохраняем изменения
                return True
            except sqlite3.IntegrityError:
                # IntegrityError возникает, если комната с таким именем уже существует (из-за UNIQUE-ограничения)
                return False
            finally:
                # Этот блок выполнится всегда — и при исключении, и в нормальной ситуации
                conn.close()
    
    def get_rooms(self) -> List[Dict]:
        """
        Получить список всех комнат.
        
        Returns:
            Список словарей с данными комнат:
            [
                {"id": 1, "name": "Room1", "camera_index": "0", "created_at": "..."},
                {"id": 2, "name": "Room2", "camera_index": "1", "created_at": "..."}
            ]
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Выбираем все записи из таблицы rooms, сортируем их по имени комнаты
        cursor.execute("SELECT * FROM rooms ORDER BY name")
        
        # Получаем все строки выборки
        rows = cursor.fetchall()
        conn.close()
        
        # Преобразуем объекты Row в словари для удобства возврата и дальнейшей работы
        return [dict(row) for row in rows]
    
    def update_person_location(self, person_id: str, room_name: str):
        """
        Обновить местоположение человека.
        
        Если человек уже существует в БД - обновляет его местоположение.
        Если не существует - создает новую запись.
        
        Args:
            person_id: ID человека (p1, p2, p3...)
            room_name: Имя комнаты, где находится человек
        """
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Проверяем, есть ли человек с таким ID уже в базе (presence-чек)
            cursor.execute("SELECT current_room FROM persons WHERE person_id = ?", (person_id,))
            existing = cursor.fetchone()
            
            if existing:
                # Если человек есть — обновляем комнату и время последнего обнаружения
                cursor.execute(
                    "UPDATE persons SET current_room = ?, last_seen = ? WHERE person_id = ?",
                    (room_name, datetime.now(), person_id)
                )
            else:
                # Если такого человека ещё не было — создаём новую запись
                cursor.execute(
                    "INSERT INTO persons (person_id, current_room, last_seen) VALUES (?, ?, ?)",
                    (person_id, room_name, datetime.now())
                )
            
            conn.commit()
            conn.close()
    
    def remove_person_from_room(self, person_id: str):
        """
        Удалить человека из комнаты (когда он исчез из кадра).
        
        ВАЖНО: Не удаляет запись о человеке, а только устанавливает current_room = NULL.
        Это позволяет сохранить ID человека для повторного использования, если он вернется.
        
        Args:
            person_id: ID человека для удаления из комнаты
        """
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Вместо удаления человека мы только сбрасываем поле current_room (NULL) и обновляем время последнего обнаружения
            cursor.execute(
                "UPDATE persons SET current_room = NULL, last_seen = ? WHERE person_id = ?",
                (datetime.now(), person_id)
            )
            
            conn.commit()
            conn.close()
    
    def find_recently_disappeared(self, room_name: str, max_seconds: int = 60) -> List[str]:
        """
        Найти недавно исчезнувших людей из комнаты.
        
        Этот метод используется для восстановления ID людей, которые недавно
        исчезли из комнаты и могут вернуться. Вместо создания нового ID,
        мы можем восстановить старый.
        
        Args:
            room_name: Имя комнаты для поиска
            max_seconds: Максимальное время с момента исчезновения (в секундах)
                        По умолчанию 60 секунд (1 минута)
        
        Returns:
            Список ID людей, которые недавно исчезли из этой комнаты
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # SQL-запрос ищет людей, у которых либо current_room пустой (т.е. человек ушёл из всех комнат),
        # либо он был именно в этой комнате, и их last_seen не старше, чем max_seconds назад
        cursor.execute("""
            SELECT person_id FROM persons 
            WHERE (current_room IS NULL OR current_room = ?)
            AND last_seen > datetime('now', '-' || ? || ' seconds')
            ORDER BY last_seen DESC
            LIMIT 10
        """, (room_name, max_seconds))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Извлекаем ID людей из результатов
        return [row['person_id'] for row in rows]
    
    def get_all_person_ids(self) -> List[str]:
        """
        Получить список всех ID людей, которые когда-либо были в системе.
        
        Используется для определения максимального ID, чтобы не создавать
        дубликаты при восстановлении трекера.
        
        Returns:
            Список всех person_id из базы данных
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        # Выбираем все идентификаторы из таблицы persons
        cursor.execute("SELECT person_id FROM persons")
        rows = cursor.fetchall()
        conn.close()
        # Возвращаем все person_id в виде списка строк
        return [row['person_id'] for row in rows]
    
    def add_movement(self, person_id: str, from_room: Optional[str], to_room: str):
        """
        Добавить запись о перемещении.
        
        Записывает в историю, что человек переместился из одной комнаты в другую.
        
        Args:
            person_id: ID человека (p1, p2, p3...)
            from_room: Откуда переместился (может быть None для первого появления)
            to_room: Куда переместился (обязательное поле)
        """
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Вставляем новую запись в таблицу movements о перемещении с указанием времени (datetime.now())
            cursor.execute(
                "INSERT INTO movements (person_id, from_room, to_room, timestamp) VALUES (?, ?, ?, ?)",
                (person_id, from_room, to_room, datetime.now())
            )
            
            conn.commit()
            conn.close()
    
    def get_movements(self, limit: int = 100) -> List[Dict]:
        """
        Получить историю перемещений.
        
        Возвращает последние перемещения, отсортированные по времени
        (самые новые первыми).
        
        Args:
            limit: Максимальное количество записей (по умолчанию 100)
            
        Returns:
            Список словарей с перемещениями:
            [
                {
                    "id": 1,
                    "timestamp": "2024-01-15 11:32:10",
                    "person_id": "p3",
                    "from_room": "Room1",
                    "to_room": "Room2"
                },
                ...
            ]
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Выбираем последние limit перемещений, отсортированные по времени убывания (сначала новые)
        cursor.execute(
            "SELECT * FROM movements ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
        
        rows = cursor.fetchall()
        conn.close()
        
        # Преобразуем Row-объекты в привычные словари
        return [dict(row) for row in rows]
    
    def get_persons_by_room(self) -> Dict[str, List[str]]:
        """
        Получить список людей по комнатам.
        
        Группирует людей по комнатам. Используется для получения
        полной картины распределения людей.
        
        Returns:
            Словарь в формате:
            {
                "Room1": ["p1", "p2", "p4"],
                "Room2": ["p3"],
                "Room3": []
            }
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Получаем всех людей и их текущие комнаты (только тех, кто сейчас присутствует в какой-либо комнате)
        cursor.execute("SELECT person_id, current_room FROM persons WHERE current_room IS NOT NULL")
        
        rows = cursor.fetchall()
        conn.close()
        
        # Группируем полученные значения по ключу 'current_room'
        result = {}
        for row in rows:
            room = row['current_room']
            person_id = row['person_id']
            
            # Если этой комнаты ещё нет в результирующем словаре — создаём для неё список
            if room not in result:
                result[room] = []
            
            # Добавляем человека в список соответствующей комнаты
            result[room].append(person_id)
        
        return result
    
    def start_room_visit(self, person_id: str, room_name: str) -> int:
        """
        Зафиксировать вход человека в комнату.
        
        Args:
            person_id: ID человека
            room_name: Имя комнаты
            
        Returns:
            ID созданной записи
        """
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Проверяем, не было ли уже активного посещения этой комнаты этим человеком (активное: exit_time IS NULL)
            cursor.execute("""
                SELECT id FROM room_visits 
                WHERE person_id = ? AND room_name = ? AND exit_time IS NULL
            """, (person_id, room_name))
            
            existing = cursor.fetchone()
            if existing:
                # Если есть активная запись (человек ещё не выходил из комнаты) — возвращаем её id, не создаём новую
                conn.close()
                return existing['id']
            
            # Создаём новую запись о посещении (человек только что вошёл в комнату)
            cursor.execute("""
                INSERT INTO room_visits (person_id, room_name, enter_time)
                VALUES (?, ?, ?)
            """, (person_id, room_name, datetime.now()))
            
            visit_id = cursor.lastrowid  # Получаем id только что созданной записи
            conn.commit()
            conn.close()
            return visit_id
    
    def end_room_visit(self, person_id: str, room_name: str = None) -> bool:
        """
        Зафиксировать выход человека из комнаты.
        
        Args:
            person_id: ID человека
            room_name: Имя комнаты (опционально, если None - завершает все активные посещения)
            
        Returns:
            True если посещение было завершено
        """
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            exit_time = datetime.now()  # Фиксируем момент выхода
            
            if room_name:
                # Если известна комната — завершаем посещение только этой комнаты этим человеком
                cursor.execute("""
                    SELECT id, enter_time FROM room_visits 
                    WHERE person_id = ? AND room_name = ? AND exit_time IS NULL
                """, (person_id, room_name))
            else:
                # Если комната не указана, завершаем все активные посещения этого человека (на случай сбоя или переезда по нескольким комнатам)
                cursor.execute("""
                    SELECT id, enter_time, room_name FROM room_visits 
                    WHERE person_id = ? AND exit_time IS NULL
                """, (person_id,))
            
            visits = cursor.fetchall()
            
            if not visits:
                # Если не найдено ни одного активного посещения — возвращаем False
                conn.close()
                return False
            
            # Для всех найденных посещений вычисляем длительность пребывания и записываем время выхода
            for visit in visits:
                enter_time_str = visit['enter_time']
                # Проверяем — enter_time приходит в виде строки или datetime
                if isinstance(enter_time_str, str):
                    # В SQLite TIMESTAMP может читаться строкой: переводим в datetime
                    enter_time = datetime.fromisoformat(enter_time_str.replace('Z', '+00:00'))
                else:
                    enter_time = enter_time_str
                
                # Вычисляем длительность пребывания в комнате в минутах
                duration = (exit_time - enter_time).total_seconds() / 60.0
                
                # Обновляем запись: ставим время выхода и длительность
                cursor.execute("""
                    UPDATE room_visits 
                    SET exit_time = ?, duration_min = ?
                    WHERE id = ?
                """, (exit_time, duration, visit['id']))
            
            conn.commit()
            conn.close()
            return True
    
    def get_active_visits(self, room_name: str = None) -> List[Dict]:
        """
        Получить активные посещения (люди, которые сейчас в комнатах).
        
        Args:
            room_name: Имя комнаты (если None - все комнаты)
            
        Returns:
            Список активных посещений с информацией о времени пребывания
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Строим корректный SQL-запрос: если указана комната — фильтруем, иначе все посещения без выхода
        if room_name:
            cursor.execute("""
                SELECT person_id, room_name, enter_time,
                       (julianday('now') - julianday(enter_time)) * 24 * 60 as duration_min
                FROM room_visits 
                WHERE room_name = ? AND exit_time IS NULL
                ORDER BY enter_time
            """, (room_name,))
        else:
            cursor.execute("""
                SELECT person_id, room_name, enter_time,
                       (julianday('now') - julianday(enter_time)) * 24 * 60 as duration_min
                FROM room_visits 
                WHERE exit_time IS NULL
                ORDER BY enter_time
            """)
        
        rows = cursor.fetchall()
        conn.close()
        # Возвращаем список словарей: каждый — активное посещение
        return [dict(row) for row in rows]
    
    def get_room_statistics(self, room_name: str, hours: int = 24) -> Dict:
        """
        Получить статистику по комнате за последние N часов.
        
        Args:
            room_name: Имя комнаты
            hours: Количество часов для анализа
            
        Returns:
            Словарь со статистикой
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Считаем статистику только за последние N часов — определяем момент отсечки
        time_threshold = datetime.now() - timedelta(hours=hours)
        
        # 1. Считаем общее количество посещений этой комнаты с момента отсечки
        cursor.execute("""
            SELECT COUNT(*) as total_visits 
            FROM room_visits 
            WHERE room_name = ? AND enter_time > ?
        """, (room_name, time_threshold))
        total_visits = cursor.fetchone()['total_visits']
        
        # 2. Считаем количество уникальных людей, которые были в этой комнате за период
        cursor.execute("""
            SELECT COUNT(DISTINCT person_id) as unique_people 
            FROM room_visits 
            WHERE room_name = ? AND enter_time > ?
        """, (room_name, time_threshold))
        unique_people = cursor.fetchone()['unique_people']
        
        # 3. Вычисляем среднюю длительность пребывания (duration_min)
        cursor.execute("""
            SELECT AVG(duration_min) as avg_duration 
            FROM room_visits 
            WHERE room_name = ? AND enter_time > ? AND duration_min IS NOT NULL
        """, (room_name, time_threshold))
        avg_duration = cursor.fetchone()['avg_duration'] or 0
        
        conn.close()
        
        # Округляем среднюю длительность до двух знаков после запятой, если есть данные
        return {
            "total_visits": total_visits,
            "unique_people": unique_people,
            "avg_duration_min": round(avg_duration, 2) if avg_duration else 0
        }
    
    def add_group_movement(self, group_id: str, from_room: Optional[str], to_room: str, members: List[str]):
        """
        Добавить запись о перемещении группы.
        
        Args:
            group_id: ID группы (Group1, Group2, ...)
            from_room: Откуда переместилась группа
            to_room: Куда переместилась группа
            members: Список ID людей в группе
        """
        import json
        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            # Преобразуем список участников группы в JSON-строку для записи в БД
            cursor.execute("""
                INSERT INTO group_movements (group_id, from_room, to_room, members, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (group_id, from_room, to_room, json.dumps(members), datetime.now()))
            conn.commit()
            conn.close()
    
    def get_group_movements(self, limit: int = 50) -> List[Dict]:
        """
        Получить историю групповых перемещений.
        
        Args:
            limit: Максимальное количество записей
            
        Returns:
            Список групповых перемещений
        """
        import json
        conn = self.get_connection()
        cursor = conn.cursor()
        # Выбираем последние limit записей о групповых перемещениях, отсортированные по времени
        cursor.execute("""
            SELECT * FROM group_movements 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()
        
        result = []
        for row in rows:
            movement = dict(row)
            # Преобразуем JSON-строку обратно в список участников
            movement['members'] = json.loads(movement['members'])
            result.append(movement)
        
        return result