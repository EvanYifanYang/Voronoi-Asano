import matplotlib.pyplot as plt
import math

class ConstantWorkspaceAlgorithm_version_2:
    def __init__(self, polygon):
        self.polygon = polygon
        self.edges = []
        self.points = []

    def plot_partial_result(self, current_p, current_edge):
        fig, ax = plt.subplots(figsize=(8, 8))
        x_coords, y_coords = zip(*self.points)
        ax.scatter(x_coords, y_coords, color='blue', label='Points')
        polygon_x, polygon_y = zip(*self.polygon.points)
        ax.plot(list(polygon_x) + [polygon_x[0]], list(polygon_y) + [polygon_y[0]], color='black', label='Bounding Box')
        for e in self.edges:
            ax.plot([e[0][0], e[1][0]], [e[0][1], e[1][1]], color='red')
        if current_edge is not None:
            (start, end) = current_edge
            ax.plot([start[0], end[0]], [start[1], end[1]], color='purple', linewidth=2.5)
        ax.scatter([current_p[0]], [current_p[1]], color='green', label='Current Point')
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3)
        plt.show()

    def plot_final_result(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        x_coords, y_coords = zip(*self.points)
        ax.scatter(x_coords, y_coords, color='blue', label='Points')
        polygon_x, polygon_y = zip(*self.polygon.points)
        ax.plot(list(polygon_x) + [polygon_x[0]], list(polygon_y) + [polygon_y[0]], color='black', label='Bounding Box')
        for e in self.edges:
            ax.plot([e[0][0], e[1][0]], [e[0][1], e[1][1]], color='red')
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3)
        plt.show()

    def create_diagram(self, points, vis_steps=False, vis_result=True):
        self.points = points

        for i, current_point in enumerate(points):
            self.process_point(i, current_point, points, vis_steps)

        if vis_result:
            self.plot_final_result()

    def process_point(self, p_index, current_point, points, vis_steps):
        current_p_x, current_p_y = current_point

        # If-Exception: Adjust
        ray_slope_determined_point = (current_p_x + 1, current_p_y)

        iteration_number = 0
        initial_edge = (None, None)

        while True:
            iteration_number += 1

            inter_slope, intersection, q_index = self.find_closest_bisector(
                current_point, p_index, points, ray_slope_determined_point, self.polygon
            )

            start_point = None
            end_point = None

            for r_index, r in enumerate(points):
                if r_index == p_index:
                    continue

                bisector_intersection, bisector_slope = self.calculate_bisector(current_point, r)

                intersection2 = self.intersect_two_lines(inter_slope, intersection, bisector_slope, bisector_intersection)

                if intersection2 is None:
                    continue

                if inter_slope is None:
                    if intersection2[1] > intersection[1]:
                        if end_point is None or intersection2[1] < end_point[1]:
                            end_point = intersection2
                    else:
                        if start_point is None or intersection2[1] > start_point[1]:
                            start_point = intersection2
                else:
                    if intersection2[0] > intersection[0]:
                        if end_point is None or intersection2[0] < end_point[0]:
                            end_point = intersection2
                    else:
                        if start_point is None or intersection2[0] > start_point[0]:
                            start_point = intersection2

            for edge in self.polygon.edges:
                bound_intersection = self.intersect_line_and_segment(intersection, inter_slope, edge)

                if bound_intersection is None:
                    continue

                if inter_slope is None:
                    if bound_intersection[1] > intersection[1]:
                        if end_point is None or bound_intersection[1] < end_point[1]:
                            end_point = bound_intersection
                    else:
                        if start_point is None or bound_intersection[1] > start_point[1]:
                            start_point = bound_intersection
                else:
                    if bound_intersection[0] > intersection[0]:
                        if end_point is None or bound_intersection[0] < end_point[0]:
                            end_point = bound_intersection
                    else:
                        if start_point is None or bound_intersection[0] > start_point[0]:
                            start_point = bound_intersection

            final_edge = self.test_orientation(current_point, start_point, end_point)

            if iteration_number == 1:
                initial_edge = final_edge
                ray_slope_determined_point = self.update_ray_slope_and_determined_point(current_point,final_edge)
            else:
                # If-Exception: Adjust
                if self.are_points_close(final_edge[1], initial_edge[0]):
                    if q_index is None or p_index < q_index:
                        self.edges.append(final_edge)
                    if vis_steps:
                        self.plot_partial_result(current_point, final_edge)
                    break
                else:
                    ray_slope_determined_point = self.update_ray_slope_and_determined_point(current_point, final_edge)

            if q_index is None or p_index < q_index:
                self.edges.append(final_edge)

            if vis_steps:
                self.plot_partial_result(current_point, final_edge)

    def find_closest_bisector(self, current_point, p_index, points, ray_slope_determined_point, polygon):
        min_distance = float('inf')
        q_index = None
        intersection = None
        inter_slope = None

        for i, point in enumerate(points):
            if i == p_index:
                continue

            current_intersection, current_bisector_slope = self.calculate_intersection(current_point, point, ray_slope_determined_point)

            if current_intersection is None:
                continue

            distance = self.calculate_distance(current_point, current_intersection)

            if distance < min_distance:
                min_distance = distance
                q_index = i
                intersection = current_intersection
                inter_slope = current_bisector_slope

        for edge in polygon.edges:
            current_intersection, current_bisector_slope = self.intersect_ray_and_segment(current_point, ray_slope_determined_point, edge)

            if current_intersection is None:
                continue

            distance = self.calculate_distance(current_point, current_intersection)

            if distance < min_distance:
                min_distance = distance
                q_index = None
                intersection = current_intersection
                inter_slope = current_bisector_slope

        ray_slope = self.calculate_slope(current_point, ray_slope_determined_point)
        intersection = self.intersect_two_lines(inter_slope, intersection, ray_slope, current_point)

        return inter_slope, intersection, q_index

    def calculate_intersection(self, current_point, point, ray_slope_determined_point, tol=1e-9):
        x0, y0 = current_point
        x1, y1 = point
        x2, y2 = ray_slope_determined_point

        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        if x1 == x0:
            bisector_slope = 0
        elif y1 == y0:
            bisector_slope = None
        else:
            original_slope = (y1 - y0) / (x1 - x0)
            bisector_slope = -1 / original_slope

        if x2 == x0:
            ray_slope = float('inf')
        else:
            ray_slope = (y2 - y0) / (x2 - x0)

        if bisector_slope is None:
            inter_x = mid_x
            if ray_slope == float('inf'):
                return None, None
            else:
                inter_y = ray_slope * (inter_x - x0) + y0
        elif ray_slope == float('inf'):
            inter_x = x0
            inter_y = bisector_slope * (inter_x - mid_x) + mid_y
        else:
            denominator = bisector_slope - ray_slope
            if abs(denominator) < tol:
                return None, None
            inter_x = (bisector_slope * mid_x - ray_slope * x0 + y0 - mid_y) / denominator
            inter_y = ray_slope * (inter_x - x0) + y0

        dir_x = x2 - x0
        dir_y = y2 - y0
        inter_dir_x = inter_x - x0
        inter_dir_y = inter_y - y0

        dot_product = dir_x * inter_dir_x + dir_y * inter_dir_y
        if dot_product < 0:
            return None, None

        return (inter_x, inter_y), bisector_slope

    def calculate_distance(self, current_point, intersection):
        return math.sqrt((current_point[0] - intersection[0]) ** 2 + (current_point[1] - intersection[1]) ** 2)

    def intersect_ray_and_segment(self, ray_start, ray_point, segment, tol=1e-9):
        (x1, y1) = ray_start
        (x2, y2) = ray_point
        (x3, y3), (x4, y4) = segment

        dx_ray = x2 - x1
        dy_ray = y2 - y1

        dx_seg = x4 - x3
        dy_seg = y4 - y3

        denominator = dx_ray * dy_seg - dy_ray * dx_seg
        if abs(denominator) < tol:
            return None, None

        t = ((x3 - x1) * dy_seg - (y3 - y1) * dx_seg) / denominator
        u = ((x3 - x1) * dy_ray - (y3 - y1) * dx_ray) / denominator

        if t < 0:
            return None, None
        if not (0 <= u <= 1):
            return None, None

        inter_x = x1 + t * dx_ray
        inter_y = y1 + t * dy_ray

        if abs(x4 - x3) < tol:
            bisector_slope = None
        elif abs(y4 - y3) < tol:
            bisector_slope = 0
        else:
            bisector_slope = (y4 - y3) / (x4 - x3)

        return (inter_x, inter_y), bisector_slope

    def calculate_bisector(self, point1, point2, tol=1e-9):
        x0, y0 = point1
        x1, y1 = point2

        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2

        if abs(x1 - x0) < tol:
            bisector_slope = 0
        elif abs(y1 - y0) < tol:
            bisector_slope = None
        else:
            original_slope = (y1 - y0) / (x1 - x0)
            bisector_slope = -1 / original_slope

        return (mid_x, mid_y), bisector_slope

    def intersect_two_lines(self, slope1, p1, slope2, p2, tol=1e-9):
        x1, y1 = p1
        x2, y2 = p2

        if slope1 is None and slope2 is None:
            return None

        if slope1 is None:
            inter_x = x1
            inter_y = slope2 * (inter_x - x2) + y2
            return (inter_x, inter_y)

        if slope2 is None:
            inter_x = x2
            inter_y = slope1 * (inter_x - x1) + y1
            return (inter_x, inter_y)

        if abs(slope1 - slope2) < tol:
            return None

        inter_x = (slope1 * x1 - slope2 * x2 + y2 - y1) / (slope1 - slope2)
        inter_y = slope1 * (inter_x - x1) + y1

        return (inter_x, inter_y)

    def intersect_line_and_segment(self, line_point, line_slope, segment, tol=1e-9):
        (x1, y1) = line_point
        (x3, y3), (x4, y4) = segment

        if line_slope is None:
            inter_x = x1
            if abs(x4 - x3) < tol:
                return None
            else:
                seg_slope = (y4 - y3) / (x4 - x3) if abs(x4 - x3) > tol else None
                seg_intercept = y3 - seg_slope * x3 if seg_slope is not None else None
                inter_y = seg_slope * inter_x + seg_intercept
        else:
            if abs(x4 - x3) < tol:
                inter_x = x3
                inter_y = line_slope * (inter_x - x1) + y1
            else:
                seg_slope = (y4 - y3) / (x4 - x3)
                if abs(seg_slope - line_slope) < tol:
                    return None
                seg_intercept = y3 - seg_slope * x3
                line_intercept = y1 - line_slope * x1
                inter_x = (seg_intercept - line_intercept) / (line_slope - seg_slope)
                inter_y = line_slope * (inter_x - x1) + y1

        if min(x3, x4) - tol <= inter_x <= max(x3, x4) + tol and min(y3, y4) - tol <= inter_y <= max(y3, y4) + tol:
            return (inter_x, inter_y)
        else:
            return None

    def test_orientation(self, current_point, start_point, end_point, tol=1e-9):
        def orientation(p, q, r):
            return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

        o1 = orientation(current_point, start_point, end_point)
        o2 = orientation(start_point, end_point, current_point)
        o3 = orientation(end_point, current_point, start_point)

        if o1 < -tol and o2 < -tol and o3 < -tol:
            return (start_point, end_point)
        else:
            return (end_point, start_point)

    def update_ray_slope_and_determined_point(self, current_point, final_edge, tol = 1e-9):
        current_point_x, current_point_y = current_point
        (start_point_x, start_point_y), (end_point_x, end_point_y) = final_edge

        temp = 0.000001

        if abs(start_point_x - end_point_x) < tol and abs(start_point_y - end_point_y) < tol:
            temp = 0.1
        if current_point_x == end_point_x:
            if end_point_y < current_point_y:
                determined_point_x = end_point_x - temp
            else:
                determined_point_x = end_point_x + temp
            determined_point_y = end_point_y
            return (determined_point_x, determined_point_y)
        elif (current_point_x < end_point_x and current_point_y < end_point_y) or (current_point_x < end_point_x and current_point_y > end_point_y):
            ray_slope = (current_point_y - end_point_y) / (current_point_x - end_point_x)
            ray_slope -= temp
            determined_point_x = current_point_x + 1
            determined_point_y = ray_slope * (determined_point_x - current_point_x) + current_point_y
            return (determined_point_x, determined_point_y)
        elif (current_point_x > end_point_x and current_point_y > end_point_y) or (current_point_x > end_point_x and current_point_y < end_point_y):
            ray_slope = (current_point_y - end_point_y) / (current_point_x - end_point_x)
            ray_slope -= temp
            determined_point_x = current_point_x - 1
            determined_point_y = ray_slope * (determined_point_x - current_point_x) + current_point_y
            return (determined_point_x, determined_point_y)
        elif current_point_y == end_point_y and current_point_x < end_point_x:
            ray_slope = -temp
            determined_point_x = current_point_x + 1
            determined_point_y = ray_slope * (determined_point_x - current_point_x) + current_point_y
            return (determined_point_x, determined_point_y)
        elif current_point_y == end_point_y and current_point_x > end_point_x:
            ray_slope = -temp
            determined_point_x = current_point_x - 1
            determined_point_y = ray_slope * (determined_point_x - current_point_x) + current_point_y
            return (determined_point_x, determined_point_y)

    def are_points_close(self, p1, p2, abs_tol=1e-9, rel_tol=1e-12):
        if p1 is None or p2 is None:
            return False
        return (math.isclose(p1[0], p2[0], abs_tol=abs_tol, rel_tol=rel_tol) and
                math.isclose(p1[1], p2[1], abs_tol=abs_tol, rel_tol=rel_tol))

    def calculate_slope(self, p1, p2, tol=1e-9):
        x0, y0 = p1
        x1, y1 = p2

        if abs(x1 - x0) < tol:
            return None
        else:
            return (y1 - y0) / (x1 - x0)