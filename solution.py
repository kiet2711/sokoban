from collections import deque
from heapq import heappop, heappush, heapify
from itertools import count
from time import perf_counter

from sokoban import sokoban_goal_state


INF = float("inf")
DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))
MAX_STATES = 100000


def _assignment_cost(costs):
    if not costs:
        return 0
    if len(costs) > len(costs[0]):
        return INF
    if len(costs) > 8 or len(costs[0]) > 8:
        return sum(min(row) for row in costs)

    # Xu ly thung co it lua chon truoc; thu dich gan truoc.
    rows = sorted(costs, key=lambda row: sum(value < INF for value in row))
    choices = [sorted((cost, goal) for goal, cost in enumerate(row)
                      if cost < INF) for row in rows]
    if any(not row for row in choices):
        return INF
    lower = [0] * (len(rows) + 1)
    for i in range(len(rows) - 1, -1, -1):
        lower[i] = lower[i + 1] + choices[i][0][0]
    best = INF
    used = set()

    def visit(i, total):
        nonlocal best
        if total + lower[i] >= best:
            return
        if i == len(rows):
            best = total
            return
        for cost, goal in choices[i]:
            if goal not in used:
                used.add(goal)
                visit(i + 1, total + cost)
                used.remove(goal)

    visit(0, 0)
    return best


def _state_key(state):
    return tuple(sorted(state.robots)), state.boxes


def solve(initial_state, timebound=120):
    started = perf_counter()
    if timebound <= 0:
        return False
    if sokoban_goal_state(initial_state):
        return initial_state
    if (len(initial_state.boxes) > len(initial_state.storage)
            or not initial_state.robots):
        return False
    deadline = started + timebound - min(2.0, 0.15 * timebound)
    floor = {(x, y) for y in range(initial_state.height)
             for x in range(initial_state.width)
             if (x, y) not in initial_state.obstacles}
    goals = tuple(sorted(initial_state.storage))
    # Ung dung BFS: tu dich di nguoc cac lan day, tam bo qua thung/robot.
    # Neu day thung tu A sang B thi robot can dung o phia sau A.
    # Vi vay, khi di nguoc B -> A, ca A va o phia sau A phai la san.
    distances = []
    for goal in goals:
        distance, queue = {goal: 0}, deque([goal])
        while queue:
            if perf_counter() >= deadline:
                return False
            point = queue.popleft()
            for dx, dy in DIRECTIONS:
                target = (point[0] + dx, point[1] + dy)
                support = (target[0] + dx, target[1] + dy)
                if target in floor and support in floor and target not in distance:
                    distance[target] = distance[point] + 1
                    queue.append(target)
        distances.append(distance)
    rows = {point: tuple(distance.get(point, INF) for distance in distances)
            for point in floor}

    def deadlocked(boxes):
        for x, y in boxes:
            if (x, y) in initial_state.storage:
                continue
            # Thung ngoai dich bi kep boi tuong theo ca hai truc.
            vertical = (x, y - 1) not in floor or (x, y + 1) not in floor
            horizontal = (x - 1, y) not in floor or (x + 1, y) not in floor
            if vertical and horizontal:
                return True
            # Tren bien ban choi, khong the day thung roi khoi bien do.
            if x in (0, initial_state.width - 1) and not any(gx == x for gx, gy in goals):
                return True
            if y in (0, initial_state.height - 1) and not any(gy == y for gx, gy in goals):
                return True
            # Khoi 2x2 lap kin boi thung/tuong thi khong thung nao thoat ra.
            for left in (x - 1, x):
                for top in (y - 1, y):
                    square = ((left, top), (left + 1, top),
                              (left, top + 1), (left + 1, top + 1))
                    if all(point not in floor or point in boxes for point in square):
                        return True
        return False

    cache = {}

    def heuristic(state):
        boxes = state.boxes
        box_cost = cache.get(boxes)
        if box_cost is None:
            if deadlocked(boxes):
                box_cost = INF
            else:
                box_cost = _assignment_cost([rows[box] for box in sorted(boxes)])
            if len(cache) >= 30000:
                cache.clear()
            cache[boxes] = box_cost
        if box_cost == INF or box_cost == 0:
            return box_cost
        # Can di toi canh it nhat mot thung truoc lan day dau tien.
        walking = min(abs(rx - bx) + abs(ry - by) - 1
                      for rx, ry in state.robots for bx, by in boxes)
        return box_cost + max(0, walking)

    initial_h = heuristic(initial_state)
    if initial_h == INF:
        return False
    serial = count()
    weights = (4.0, 3.0, 2.0, 1.5, 1.0)
    stage, weight = 0, weights[0]
    frontier = [(weight * initial_h, initial_h, next(serial), initial_state)]
    best_g = {_state_key(initial_state): initial_state.gval}
    answer, bound = False, INF

    while frontier and perf_counter() < deadline:
        _, h, _, state = heappop(frontier)
        g = state.gval
        if best_g.get(_state_key(state)) != g or g + h >= bound:
            continue
        if sokoban_goal_state(state):
            answer, bound = state, g
            stage += 1
            weight = weights[min(stage, len(weights) - 1)]
            # Loai nhanh khong the cai thien, roi ha trong so de tim ky hon.
            frontier = [(entry[3].gval + weight * entry[1], entry[1], entry[2], entry[3])
                        for entry in frontier
                        if entry[3].gval == best_g.get(_state_key(entry[3]))
                        and entry[3].gval + entry[1] < bound]
            heapify(frontier)
            continue
        if len(best_g) >= MAX_STATES:
            # Giai phong bo nho tim kiem, giu lai answer va tim tiep.
            stage += 1
            weight = (weights[stage % len(weights)] if answer is not False
                      else 4.0 + 2.0 * min(stage, 8))
            frontier = [(weight * initial_h, initial_h, next(serial), initial_state)]
            best_g = {_state_key(initial_state): initial_state.gval}
            continue
        for child in state.successors():
            if perf_counter() >= deadline:
                break
            if child.boxes == state.boxes:
                child.boxes = state.boxes  # Tai su dung frozenset bat bien.
            key = _state_key(child)
            if child.gval >= bound or child.gval >= best_g.get(key, INF):
                continue
            child_h = heuristic(child)
            if child.gval + child_h >= bound:
                continue
            best_g[key] = child.gval
            heappush(frontier, (child.gval + weight * child_h,
                                child_h, next(serial), child))
    return answer
