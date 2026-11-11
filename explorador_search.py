# 1. CLASE DEL PROBLEMA (ESTRUCTURA BASE AIMA Y FORMALIZACIÓN DEL EXPLORADOR)
import time
import heapq
from collections import deque


class Problem:
    """Clase abstracta base del estándar AIMA (Russell & Norvig) para problemas de búsqueda."""

    def __init__(self, initial, goal=None):
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        raise NotImplementedError

    def result(self, state, action):
        raise NotImplementedError

    def goal_test(self, state):
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        return c + 1


class Node:
    """Estructura de nodo en el árbol de búsqueda según AIMA-Python."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0 if parent is None else parent.depth + 1

    def expand(self, problem):
        return [
            self.child_node(problem, action)
            for action in problem.actions(self.state)
        ]

    def child_node(self, problem, action):
        next_state = problem.result(self.state, action)
        next_cost = problem.path_cost(
            self.path_cost, self.state, action, next_state
        )
        return Node(next_state, self, action, next_cost)

    def solution(self):
        return [node.action for node in self.path()[1:]]

    def path(self):
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))


class JuegoExplorador(Problem):
    """Formalización del Caso B: El juego del explorador en una cuadrícula 10x10."""

    MOVIMIENTOS = {
        "Arriba": (-1, 0),
        "Abajo": (1, 0),
        "Izquierda": (0, -1),
        "Derecha": (0, 1),
    }

    def __init__(self, grid, initial=(0, 0), goal=(9, 9)):
        super().__init__(initial, goal)
        self.grid = grid
        self.filas = len(grid)
        self.columnas = len(grid[0])

    def actions(self, state):
        """2. ACTIONS(s): Retorna las acciones válidas dentro de límites y sin muros (0)."""
        x, y = state
        acciones_validas = []
        for accion, (dx, dy) in self.MOVIMIENTOS.items():
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < self.filas
                and 0 <= ny < self.columnas
                and self.grid[nx][ny] == 0
            ):
                acciones_validas.append(accion)
        return acciones_validas

    def result(self, state, action):
        """3. RESULT(s, a): Modelo de transición determinista."""
        x, y = state
        dx, dy = self.MOVIMIENTOS[action]
        return (x + dx, y + dy)

    def goal_test(self, state):
        """4. IS-GOAL(s): Verifica si el explorador alcanzó la coordenada meta."""
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """5. c(s, a, s'): Costo uniforme de 1 unidad por cada paso ortogonal."""
        return c + 1

    def h(self, node):
        """Heurística admisible h(n): Distancia Manhattan hasta la meta."""
        x, y = node.state
        gx, gy = self.goal
        return abs(gx - x) + abs(gy - y)


def bfs(problem):
    """Búsqueda en Anchura (Breadth-First Search) - Frontera FIFO."""
    frontera = deque([Node(problem.initial)])
    visitados = {problem.initial}
    nodos_expandidos = 0

    while frontera:
        nodo = frontera.popleft()
        if problem.goal_test(nodo.state):
            return nodo, nodos_expandidos

        nodos_expandidos += 1
        for hijo in nodo.expand(problem):
            if hijo.state not in visitados:
                visitados.add(hijo.state)
                frontera.append(hijo)
    return None, nodos_expandidos


def dfs(problem):
    """Búsqueda en Profundidad (Depth-First Search) - Frontera LIFO."""
    frontera = [Node(problem.initial)]
    visitados = set()
    nodos_expandidos = 0

    while frontera:
        nodo = frontera.pop()
        if nodo.state in visitados:
            continue
        visitados.add(nodo.state)

        if problem.goal_test(nodo.state):
            return nodo, nodos_expandidos

        nodos_expandidos += 1
        for hijo in reversed(nodo.expand(problem)):
            if hijo.state not in visitados:
                frontera.append(hijo)
    return None, nodos_expandidos


def astar_search(problem):
    """Búsqueda Informada A* con función de evaluación f(n) = g(n) + h(n)."""
    nodo_inicial = Node(problem.initial)
    frontera = []
    contador = 0
    heapq.heappush(
        frontera,
        (
            problem.h(nodo_inicial),
            contador,
            nodo_inicial,
        ),
    )
    mejor_costo = {problem.initial: 0}
    nodos_expandidos = 0

    while frontera:
        _, _, nodo = heapq.heappop(frontera)

        if problem.goal_test(nodo.state):
            return nodo, nodos_expandidos

        if nodo.path_cost > mejor_costo.get(nodo.state, float("inf")):
            continue

        nodos_expandidos += 1
        for hijo in nodo.expand(problem):
            nuevo_costo = hijo.path_cost
            if nuevo_costo < mejor_costo.get(hijo.state, float("inf")):
                mejor_costo[hijo.state] = nuevo_costo
                contador += 1
                f_n = nuevo_costo + problem.h(hijo)
                heapq.heappush(frontera, (f_n, contador, hijo))
    return None, nodos_expandidos


if __name__ == "__main__":
    GRID = [
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
        [1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    ]

    problema = JuegoExplorador(GRID, initial=(0, 0), goal=(9, 9))
    costo_optimo = 18

    algoritmos = [
        ("BFS (Búsqueda en Anchura)", bfs),
        ("DFS (Búsqueda en Profundidad)", dfs),
        ("A* (Búsqueda Informada)", astar_search),
    ]

    print(
        f"{'Algoritmo de Búsqueda':<32} | {'Nodos Exp.':<10} | {'Tiempo (ms)':<12} | {'Costo g(n)':<10} | {'¿Óptimo?'}"
    )
    print("-" * 82)

    for nombre, funcion in algoritmos:
        inicio_cpu = time.perf_counter()
        solucion, expandidos = funcion(problema)
        tiempo_ms = (time.perf_counter() - inicio_cpu) * 1000
        costo = solucion.path_cost if solucion else -1
        es_optimo = "Sí" if costo == costo_optimo else "No"
        print(
            f"{nombre:<32} | {expandidos:<10} | {tiempo_ms:<12.4f} | {costo:<10} | {es_optimo}"
        )
