import random
import math

def is_collinear(p1, p2, p3):
    """ 判断三点是否共线 """
    return (p2[1] - p1[1]) * (p3[0] - p1[0]) == (p3[1] - p1[1]) * (p2[0] - p1[0])

def circumcircle(p1, p2, p3):
    """ 计算三点确定的外接圆圆心和半径平方 """
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if d == 0:
        return None  # 三点共线，无外接圆
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
    r2 = (ux - ax)**2 + (uy - ay)**2  # 半径平方
    return (ux, uy, r2)

def is_cocircular(p1, p2, p3, p4):
    """ 判断四点是否共圆 """
    circle = circumcircle(p1, p2, p3)
    if circle is None:
        return False  # 如果前三点共线，则无外接圆，必不共圆
    ux, uy, r2 = circle
    return math.isclose((p4[0] - ux)**2 + (p4[1] - uy)**2, r2)

def generate_points(n, x_range, y_range):
    points = []
    point_set = set()  # 使用集合提高重复检测效率
    attempts = 0

    while len(points) < n:
        attempts += 1
        if attempts > n * 50:  # 防止死循环
            print("Warning: Unable to find enough valid points. Try reducing n or adjusting range.")
            break

        x = random.randint(x_range[0] + 1, x_range[1] - 1)
        y = random.randint(y_range[0] + 1, y_range[1] - 1)
        new_point = (x, y)

        # 确保点不重复
        if new_point in point_set:
            continue

        # 仅检查最近的几个点，加速判断
        if len(points) > 2:
            recent_points = points[-min(10, len(points)):]  # 仅检查最近的10个点
            if any(is_collinear(new_point, p1, p2) for i, p1 in enumerate(recent_points) for p2 in recent_points[i+1:]):
                continue

        if len(points) > 3:
            recent_points = points[-min(10, len(points)):]  # 仅检查最近的10个点
            if any(is_cocircular(new_point, p1, p2, p3) for i, p1 in enumerate(recent_points) for j, p2 in enumerate(recent_points[i+1:]) for p3 in recent_points[i+j+2:]):
                continue

        points.append(new_point)
        point_set.add(new_point)

    return points

random.seed(1194171)  # 固定随机种子，保证可复现
points = generate_points(3000, (-10000, 10000), (-10000, 10000))
print(f"Generated {len(points)} points")
print(points)
