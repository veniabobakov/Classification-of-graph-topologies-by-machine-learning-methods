from networkx.algorithms.components.weakly_connected import is_weakly_connected
import networkx as nx
import numpy as np
from random import randint  # импортируем Библиотеки


class Generate(object):
    """Класс генератора графов
    для вызова необходимо присвоить класс переменной.

    Чтение происходит следующим образом:

    with open(название.npy, 'rb') as f:
        a = np.load(f) <- массив ребер графа
        b = np.load(f) <- признаки графа

    Данную операцию нужно будет повторить n (количество графов) раз

    """

    # Граф сеть малого мира
    def smallworld(self, n: int, nodes: int, k: float, p: float, file: str):
        """
        n - количество графов  \n
        nodes - количество вершин в графе \n
        k - ко скольким соседним вершинам присоединяется вершина \n
        p - вероятность добавить новое ребро

        """
        with open(file, 'wb') as f:
            for i in range(n):  # цикл для генерация n сетей
                watts_strogatz = nx.newman_watts_strogatz_graph(nodes, k, p)  # генерируем сеть малого мира
                np.save(f, nx.edges(watts_strogatz))  # сохраняем numpy массив пары вершин
                np.save(f, np.array([nx.average_shortest_path_length(watts_strogatz),
                                     nx.average_clustering(watts_strogatz)]))  # сохраняем numpy характеристики сети

    # Случайны Граф
    def rand(self, n: int, nodes: int, edges: int, file: str):
        """
        n - количество графов \n
        nodes - количество вершин в графе \n
        edges - количество ребер  \n

        """
        m = 0
        with open(file, 'wb') as f:  # генерируем n сетей
            while m < n:
                try:
                    r = nx.gnm_random_graph(nodes, edges)  # гененрируем рандомный граф
                    np.save(f, nx.edges(r))  # сохраняем numpy массив пары вершин
                    np.save(f, np.array([nx.average_shortest_path_length(r),
                                         nx.average_clustering(r)]))  # сохраняем numpy характеристики сети
                    m += 1
                except:  # если граф получился несвязным повторяем итерацию
                    continue

    # Безмасштабный граф
    def scale_free_graph(self, n: int, nodes: int, alpha: float, beta: float, gamma: float, file: str):
        """
        n - количество графов \n
        nodes - количество вершин в графе \n
        alpha - вероятность добавить вершину к существующей вершине in-degree distribution. \n
        beta - вероятность добавить ребро между вершинами. \n
        gamma - вероятность добавить вершину к существующей вершине out-degree distribution.

        """
        m = 0
        with open(file, 'wb') as f:  # генерируем n сетей
            while m < n:
                try:
                    scale_free = nx.scale_free_graph(nodes, alpha, beta, gamma)  # гененрируем scale free
                    np.save(f, nx.edges(scale_free))  # сохраняем numpy массив пары вершин
                   # сохраняем numpy характеристики сети
                    m += 1
                except:  # если граф получился несвязным повторяем итерацию
                    continue
# Случайный граф малого мира
    def rnd_smallworld(self, n: int, min_nodes: int, max_nodes: int, file: str):
        """
        n - количество графов \n
        min_nodes - минимальное количество вершин в графе \n
        max_nodes - максимальноеколичество вершин в графе \n

        """
        for _ in range(n):
            with open(file, 'ab') as f:  #######################!!!!!!!!!!!!!!!!! ТУТ !!!!!!!! #############
                watts_strogatz = nx.newman_watts_strogatz_graph(randint(min_nodes, max_nodes), randint(2, 4), float(
                    randint(10, 99)) / 100)  # генерируем сеть малого мира)
                np.save(f, nx.edges(watts_strogatz))  # сохраняем numpy массив пары вершин
                np.save(f, np.array([0]))

    # Случайный граф со случайными параметрами
    def rnd_rand(self, n: int, min_nodes: int, max_nodes: int, min_edges: int, max_edges: int, file: str):
        """
        n - количество графов \n
        min_nodes - минимальное количество вершин в графе \n
        max_nodes - максимальноеколичество вершин в графе \n
        min_edges - минимальное число ребер в графе \n
        max_edges - максимальное число вершин в графе

        """
        m = 0
        with open(file, 'ab') as f:  # генерируем n сетей
            while m < n:
                r = nx.gnm_random_graph(randint(min_nodes, max_nodes),
                                            randint(min_edges, max_edges))  # гененрируем рандомный граф
                if nx.is_connected(r):
                    np.save(f, nx.edges(r))  # сохраняем numpy массив пары вершин
                    np.save(f, np.array([1]))
                    m += 1
                    print(m)
                else:
                    continue

    # Случайный безмасштабный граф
    def rnd_scale_free_graph(self, n: int, min_nodes: int, max_nodes: int, file: str):
        """
        n - количество графов \n
        min_nodes - минимальное количество вершин в графе \n
        max_nodes - максимальноеколичество вершин в графе \n
        a, b (необязательно) - параметры для выбора вероятности

        """
        m = 0
        with open(file, 'ab') as f:  # генерируем n сетей
            m = 0
            while m < n:  # генерируем n сетей
                # Генерация вероятностей alpha, beta, gamma
                val = []
                num1 = randint(10, 99)
                num2 = randint(1, num1 - 1)
                val.append(num2)
                val.append(num1 - num2)
                val.append(100 - num1)
                gamma = min(val)
                beta = max(val)
                val.pop(val.index(gamma))
                val.pop(val.index(beta))
                alpha = val[0]
                scale_free = nx.scale_free_graph(randint(min_nodes, max_nodes), alpha / 100, beta / 100,
                                                     gamma / 100)  # гененрируем scale free
                if nx.is_weakly_connected(scale_free):
                    np.save(f, list(nx.edges(scale_free)))  # сохраняем numpy массив пары вершин
                    np.save(f, np.array([2]) ) # сохраняем numpy характеристики сети
                    m += 1
                    print(m)
                else:
                    continue
    def rnd_regular(self, n: int, min_nodes: int , max_nodes: int, file: str):
        with open(file, 'ab') as f:
            k = 0
            while k < n:
                while True:
                    nodes = randint(min_nodes, max_nodes)
                    d = randint(2, 8)
                    if (nodes * d) % 2 == 0:
                        break
                    else: nodes += 1; break
                r = nx.random_regular_graph(d,nodes)
                if nx.number_connected_components(r) and nx.is_connected(r):
                    r = nx.random_regular_graph(d,nodes)
                    np.save(f, list(nx.edges(r)))  # сохраняем numpy массив пары вершин
                    np.save(f, np.array([3]) ) # сохраняем numpy характеристики сети
                    np.save(f, np.array([nodes]))
                    print(k)
                    k += 1