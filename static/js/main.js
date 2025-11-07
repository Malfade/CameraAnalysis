/**
 * JavaScript для обновления данных на dashboard
 */

// Форматирование времени
function formatDuration(minutes) {
    if (!minutes) return '0 мин';
    const mins = Math.floor(minutes);
    const hours = Math.floor(mins / 60);
    const minsRemainder = mins % 60;
    
    if (hours > 0) {
        return `${hours}ч ${minsRemainder}мин`;
    }
    return `${mins}мин`;
}

// Обновление статистики
function updateStatistics() {
    fetch('/api/rooms')
        .then(response => response.json())
        .then(data => {
            // Подсчитываем общую статистику
            let totalPeople = 0;
            let totalRooms = Object.keys(data).length;
            
            Object.values(data).forEach(room => {
                totalPeople += room.count || 0;
            });
            
            document.getElementById('total-rooms').textContent = totalRooms;
            document.getElementById('total-people').textContent = totalPeople;
        })
        .catch(error => {
            console.error('Ошибка при загрузке статистики:', error);
        });
}

// Обновление статуса комнат
function updateRoomsStatus() {
    fetch('/api/rooms')
        .then(response => response.json())
        .then(data => {
            const container = document.getElementById('rooms-status');
            container.innerHTML = '';
            
            if (Object.keys(data).length === 0) {
                container.innerHTML = '<div class="col-12"><div class="empty-state">Нет данных о комнатах</div></div>';
                return;
            }
            
            Object.keys(data).forEach(roomName => {
                const roomData = data[roomName];
                const count = roomData.count || 0;
                const persons = roomData.persons || [];
                
                // Проверяем, есть ли класс security-page
                const isSecurityPage = document.body.classList.contains('security-page');
                
                const roomCard = document.createElement('div');
                if (isSecurityPage) {
                    roomCard.className = 'room-card-security';
                    roomCard.innerHTML = `
                        <div class="room-name">${roomName}</div>
                        <div class="room-count">${count}</div>
                        <div class="room-persons">
                            ${persons.length > 0 
                                ? 'ID: ' + persons.join(', ')
                                : 'NO SUBJECTS'
                            }
                        </div>
                    `;
                } else {
                    roomCard.className = 'col-md-6 mb-3';
                    roomCard.innerHTML = `
                        <div class="room-card">
                            <h3>${roomName}</h3>
                            <div class="person-count">${count}</div>
                            <div class="person-count-label">${count === 1 ? 'человек' : count < 5 ? 'человека' : 'человек'}</div>
                            <div class="person-list">
                                ${persons.length > 0 
                                    ? persons.map(p => `<span class="person-badge">${p}</span>`).join(' ')
                                    : '<span style="opacity: 0.7;">Пусто</span>'
                                }
                            </div>
                        </div>
                    `;
                }
                container.appendChild(roomCard);
            });
            
            // Обновляем время последнего обновления
            const now = new Date();
            const timeStr = now.toLocaleTimeString('ru-RU');
            document.getElementById('last-update').textContent = timeStr;
        })
        .catch(error => {
            console.error('Ошибка при загрузке данных о комнатах:', error);
        });
}

// Обновление активных посещений
function updateActiveVisits() {
    fetch('/api/active_visits')
        .then(response => response.json())
        .then(data => {
            const container = document.getElementById('active-visits-list');
            
            const isSecurityPage = document.body.classList.contains('security-page');
            
            if (!data || data.length === 0) {
                container.innerHTML = `<div class="${isSecurityPage ? 'security-loading' : 'empty-state'}">${isSecurityPage ? 'NO ACTIVE VISITS' : 'Нет активных посещений'}</div>`;
                document.getElementById('active-visits').textContent = '0';
                return;
            }
            
            document.getElementById('active-visits').textContent = data.length;
            
            container.innerHTML = '';
            data.forEach(visit => {
                const visitCard = document.createElement('div');
                const duration = formatDuration(visit.duration_min);
                
                if (isSecurityPage) {
                    visitCard.className = 'visit-item-security';
                    visitCard.innerHTML = `
                        <div class="visit-person">${visit.person_id}</div>
                        <div class="visit-room">ROOM: ${visit.room_name}</div>
                        <div class="visit-duration">DURATION: ${duration}</div>
                    `;
                } else {
                    visitCard.className = 'visit-card';
                    visitCard.innerHTML = `
                        <div class="person-id">${visit.person_id}</div>
                        <div style="font-size: 0.85rem; color: #666; margin-top: 5px;">
                            <strong>${visit.room_name}</strong>
                        </div>
                        <div class="visit-duration">
                            ⏱️ ${duration}
                        </div>
                        <div style="font-size: 0.75rem; color: #999; margin-top: 5px;">
                            Вход: ${visit.enter_time_str || visit.enter_time}
                        </div>
                    `;
                }
                container.appendChild(visitCard);
            });
        })
        .catch(error => {
            console.error('Ошибка при загрузке активных посещений:', error);
        });
}

// Обновление истории перемещений
function updateMovements() {
    fetch('/api/movements')
        .then(response => response.json())
        .then(data => {
            const tbody = document.getElementById('movements-table');
            tbody.innerHTML = '';
            
            const isSecurityPage = document.body.classList.contains('security-page');
            
            if (data.length === 0) {
                tbody.innerHTML = `<tr><td colspan="4" class="text-center ${isSecurityPage ? 'security-loading' : 'text-muted'}">${isSecurityPage ? 'NO MOVEMENTS' : 'Нет перемещений'}</td></tr>`;
                return;
            }
            
            data.forEach(movement => {
                const row = document.createElement('tr');
                const time = movement.time || movement.timestamp || 'N/A';
                const fromRoom = movement.from_room || '-';
                
                if (isSecurityPage) {
                    row.innerHTML = `
                        <td>${time}</td>
                        <td><span class="value">${movement.person_id}</span></td>
                        <td>${fromRoom}</td>
                        <td><span class="value">${movement.to_room}</span></td>
                    `;
                } else {
                    row.innerHTML = `
                        <td>${time}</td>
                        <td><span class="person-id-badge">${movement.person_id}</span></td>
                        <td>${fromRoom}</td>
                        <td><strong>${movement.to_room}</strong></td>
                    `;
                }
                tbody.appendChild(row);
            });
        })
        .catch(error => {
            console.error('Ошибка при загрузке истории перемещений:', error);
        });
}

// Обновление групповых перемещений
function updateGroupMovements() {
    fetch('/api/group_movements')
        .then(response => response.json())
        .then(data => {
            const tbody = document.getElementById('group-movements-table');
            tbody.innerHTML = '';
            
            const isSecurityPage = document.body.classList.contains('security-page');
            
            if (!data || data.length === 0) {
                tbody.innerHTML = `<tr><td colspan="5" class="text-center ${isSecurityPage ? 'security-loading' : 'text-muted'}">${isSecurityPage ? 'NO GROUP MOVEMENTS' : 'Нет групповых перемещений'}</td></tr>`;
                return;
            }
            
            data.forEach(movement => {
                const row = document.createElement('tr');
                const time = movement.time || movement.timestamp || 'N/A';
                const fromRoom = movement.from_room || '-';
                const members = movement.members ? movement.members.join(', ') : '';
                
                if (isSecurityPage) {
                    row.innerHTML = `
                        <td>${time}</td>
                        <td><span class="value">${movement.group_id}</span></td>
                        <td><span class="value">${members}</span></td>
                        <td>${fromRoom}</td>
                        <td><span class="value">${movement.to_room}</span></td>
                    `;
                } else {
                    row.innerHTML = `
                        <td>${time}</td>
                        <td><strong style="color: #667eea;">${movement.group_id}</strong></td>
                        <td>${members.split(',').map(m => `<span class="person-id-badge">${m.trim()}</span>`).join(' ')}</td>
                        <td>${fromRoom}</td>
                        <td><strong>${movement.to_room}</strong></td>
                    `;
                }
                tbody.appendChild(row);
            });
        })
        .catch(error => {
            console.error('Ошибка при загрузке групповых перемещений:', error);
        });
}