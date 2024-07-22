# -*- coding: utf-8 -*-

import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import heapq

# Створення графа
G = nx.Graph()

# Додавання вершин (станції метро)
stations = ["Теремки", "Іподром", "ВДНГ", "Васильківська", "Голосіївська", "Деміївська", "Либідська", "Палац Україна", "Олімпійська", "Площа Українських Героїв", "Майдан Незалежності", "Поштова площа", "Контрактова площа", "Тараса Шевченка", "Почайна", "Оболонь", "Мінська", "Героїв Дніпра", "Лісова", "Чернігівська", "Дарниця", "Лівобережна", "Гідропарк", "Дніпро", "Арсенальна", "Хрещатик", "Театральна", "Університет", "Вокзальна", "Політехнічний інститут", "Шулявська", "Берестейська", "Нивки", "Святошин", "Житомирська", "Академмістечко", "Сирець", "Дорогожичі", "Лук'янівська", "Золоті ворота", "Палац спорту", "Кловська" , "Печерська", "Звіринецька", "Видубичі", "Славутич", "Осокорки", "Позняки", "Харківська", "Вирлиця", "Бориспільська", "Червоний хутір"]
G.add_nodes_from(stations)

# Додавання ребер (лінії метро між станціями) з вагами
edges = [
    ("Теремки", "Іподром", 2),
    ("Іподром", "ВДНГ", 2),
    ("ВДНГ", "Васильківська", 2),
    ("Васильківська", "Голосіївська", 2),
    ("Голосіївська", "Деміївська", 2),
    ("Деміївська", "Либідська", 2),
    ("Либідська", "Палац Україна", 2),
    ("Палац Україна", "Олімпійська", 2),
    ("Олімпійська", "Площа Українських Героїв", 2),
    ("Площа Українських Героїв", "Майдан Незалежності", 2),
    ("Майдан Незалежності", "Поштова площа", 2),
    ("Поштова площа", "Контрактова площа", 2),
    ("Контрактова площа", "Тараса Шевченка", 2),
    ("Тараса Шевченка", "Почайна", 2),
    ("Почайна", "Оболонь", 2),
    ("Оболонь", "Мінська", 2),
    ("Мінська", "Героїв Дніпра", 2),
    ("Театральна", "Хрещатик", 1),
    ("Хрещатик", "Майдан Незалежності", 1),
    ("Хрещатик", "Арсенальна", 2),
    ("Арсенальна", "Дніпро", 2),
    ("Дніпро", "Гідропарк", 2),
    ("Гідропарк", "Лівобережна", 2),
    ("Лівобережна", "Дарниця", 2),
    ("Дарниця", "Чернігівська", 2),
    ("Чернігівська", "Лісова", 2),
    ("Театральна", "Університет", 1),
    ("Університет", "Вокзальна", 2),
    ("Вокзальна", "Політехнічний інститут", 2),
    ("Політехнічний інститут", "Шулявська", 2),
    ("Шулявська", "Берестейська", 2),
    ("Берестейська", "Нивки", 2),
    ("Нивки", "Святошин", 2),
    ("Святошин", "Житомирська", 2),
    ("Житомирська", "Академмістечко", 2),
    ("Сирець", "Дорогожичі", 2),
    ("Дорогожичі", "Лук'янівська", 2),
    ("Лук'янівська", "Золоті ворота", 2),
    ("Золоті ворота", "Театральна", 1),
    ("Золоті ворота", "Палац спорту", 1),
    ("Палац спорту", "Площа Українських Героїв", 1),
    ("Палац спорту", "Кловська", 1),
    ("Кловська", "Печерська", 2),
    ("Печерська", "Звіринецька", 2),
    ("Звіринецька", "Видубичі", 2),
    ("Видубичі", "Славутич", 2),
    ("Славутич", "Осокорки", 2),
    ("Осокорки", "Позняки", 2),
    ("Позняки", "Харківська", 2),
    ("Харківська", "Вирлиця", 2),
    ("Вирлиця", "Бориспільська", 2),
    ("Бориспільська", "Червоний хутір", 2),
]
G.add_weighted_edges_from(edges)

# Візуалізація графа
plt.figure(figsize=(24, 16))
nx.draw(G, with_labels=True, node_color='skyblue', node_size=300, edge_color='gray', font_size=6, font_weight='bold')
plt.title("Transport Network Graph")
plt.show()

# Аналіз основних характеристик
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
degree_centrality = nx.degree_centrality(G)

print(f"Кількість вершин: {num_nodes}")
print(f"Кількість ребер: {num_edges}")
print("Ступінь кожної вершини:")
for node, degree in degree_centrality.items():
    print(f"{node}: {degree:.2f}")

# Завдання 2

# Реалізація DFS
def dfs(graph, start):
    visited, stack = set(), [start]
    paths = {start: [start]}
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    stack.append(neighbor)
                    paths[neighbor] = paths[vertex] + [neighbor]
    return paths

# Реалізація BFS
def bfs(graph, start):
    visited, queue = set(), deque([start])
    paths = {start: [start]}
    while queue:
        vertex = queue.popleft()
        if vertex not in visited:
            visited.add(vertex)
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    queue.append(neighbor)
                    paths[neighbor] = paths[vertex] + [neighbor]
    return paths

# Виконання DFS і BFS
start_station = "Теремки"
dfs_paths = dfs(G, start_station)
bfs_paths = bfs(G, start_station)

# Порівняння результатів
print("DFS Paths:")
for station in stations:
    if station in dfs_paths:
        print(f"{start_station} -> {station}: {dfs_paths[station]}")

print("\nBFS Paths:")
for station in stations:
    if station in bfs_paths:
        print(f"{start_station} -> {station}: {bfs_paths[station]}")

# Завдання 3

# Реалізація алгоритму Дейкстри
def dijkstra(graph, start):
    queue = []
    heapq.heappush(queue, (0, start))
    distances = {node: float('inf') for node in graph.nodes}
    distances[start] = 0
    shortest_path = {node: [] for node in graph.nodes}
    
    while queue:
        current_distance, current_node = heapq.heappop(queue)
        
        if current_distance > distances[current_node]:
            continue
        
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight['weight']
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))
                shortest_path[neighbor] = shortest_path[current_node] + [current_node]
    
    return distances, shortest_path

# Виконання алгоритму Дейкстри для кожної вершини
all_distances = {}
all_paths = {}

for station in stations:
    distances, paths = dijkstra(G, station)
    all_distances[station] = distances
    all_paths[station] = paths

# Порівняння результатів для прикладу (між "Теремки" та іншими станціями)
start_station = "Теремки"
print(f"Найкоротші шляхи з {start_station}:")
for station in stations:
    if station != start_station:
        path = all_paths[start_station][station] + [station]
        print(f"{start_station} -> {station}: {path}, Відстань: {all_distances[start_station][station]}")

# Візуалізація графа
plt.figure(figsize=(24, 16))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=300, edge_color='gray', font_size=6, font_weight='bold')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title("Transport Network Graph with Weights")
plt.show()