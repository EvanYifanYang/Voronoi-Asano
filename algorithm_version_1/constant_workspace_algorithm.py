import matplotlib.pyplot as plt
import math

class ConstantWorkspaceAlgorithm_version_1:
    def __init__(self, rectangleBoundingBox):
        self.rectangleBoundingBox = rectangleBoundingBox # Bounding Box
        self.edges = [] # Output
        self.points = [] # Input

    def create_diagram(self, points, vis_steps=True, vis_result=True):
        flag = False
        self.points = points
        for i, current_p in enumerate(points):
            # print(f"\nProcessing point {current_p} (Index {i})")
            flag = self.process_point(current_p, points, i, vis_steps)
            if flag:
                break

        if flag:
            return flag

        if vis_result:
            self.plot_final_result()

    def plot_partial_result(self, current_p, final_direct_edge):
        fig, ax = plt.subplots(figsize=(8, 8))
        x_coords, y_coords = zip(*self.points)
        ax.scatter(x_coords, y_coords, color='blue', label='Points')
        rectangle_x, rectangle_y = zip(*self.rectangleBoundingBox.vertices)
        ax.plot(list(rectangle_x) + [rectangle_x[0]], list(rectangle_y) + [rectangle_y[0]], color='black', label='Bounding Box')
        for e in self.edges:
            ax.plot([e[0][0], e[1][0]], [e[0][1], e[1][1]], color='red')
        if final_direct_edge is not None:
            (start, end) = final_direct_edge
            ax.plot([start[0], end[0]], [start[1], end[1]], color='purple', linewidth=2.5)
        ax.scatter([current_p[0]], [current_p[1]], color='green', label='Current Point')
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3)
        plt.show()

    def plot_final_result(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        x_coords, y_coords = zip(*self.points)
        ax.scatter(x_coords, y_coords, color='blue', label='Points')
        rectangle_x, rectangle_y = zip(*self.rectangleBoundingBox.vertices)
        ax.plot(list(rectangle_x) + [rectangle_x[0]], list(rectangle_y) + [rectangle_y[0]], color='black', label='Bounding Box')
        for e in self.edges:
            ax.plot([e[0][0], e[1][0]], [e[0][1], e[1][1]], color='red')
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3)
        plt.show()

    def process_point(self, current_p, points, p_index, vis_steps, tol = 1e-9):
        special_flag = False
        ray_slope = 1.000 # Initial value = -1.000
        used_indices = {p_index} # Eliminate the processed point
        current_p_x, current_p_y = current_p
        ray_slope_determined_point = (current_p_x - 1, current_p_y - 1) # Determine the ray slope after the first iteration
        initial_ray_slope_direction_flag = False
        offset_flag = -1
        collisions_number = 0 # Judge if stop the current cell process
        added_edge_number = 0
        preview_edge = None
        # special_flag = False

        if (self.rectangleBoundingBox.x_range[0] < current_p_x < (
                self.rectangleBoundingBox.x_range[0] + self.rectangleBoundingBox.x_range[1]) / 2
                and (self.rectangleBoundingBox.y_range[0] + self.rectangleBoundingBox.y_range[1]) / 2 < current_p_y <
                self.rectangleBoundingBox.y_range[1]):
            ray_slope = -1.000
            ray_slope_determined_point = (current_p_x + 1, current_p_y - 1)
        elif ((self.rectangleBoundingBox.x_range[0] + self.rectangleBoundingBox.x_range[1]) / 2 < current_p_x <
              self.rectangleBoundingBox.x_range[1]
              and self.rectangleBoundingBox.y_range[0] < current_p_y < (self.rectangleBoundingBox.y_range[0] +
                   self.rectangleBoundingBox.y_range[1]) / 2):
            ray_slope = -1.000
            ray_slope_determined_point = (current_p_x - 1, current_p_y + 1)
        elif (self.rectangleBoundingBox.x_range[0]< current_p_x <
               (self.rectangleBoundingBox.x_range[0] + self.rectangleBoundingBox.x_range[1]) / 2
              and self.rectangleBoundingBox.y_range[0] < current_p_y < (self.rectangleBoundingBox.y_range[0] +
                   self.rectangleBoundingBox.y_range[1]) / 2):
            ray_slope = 1.000
            ray_slope_determined_point = (current_p_x + 1, current_p_y + 1)

        # Store the first edge to prepare for changing direction
        iterations_number = 0
        initial_edge = None

        while True:
            collisions_number_change_flag = False
            iterations_number += 1
            # print(f"Current iteration: {iterations_number}, Collisions number: {collisions_number}, Added edge number: {added_edge_number}")
            # print(f"Preview edge: {preview_edge}")
            # print(f"Current ray slope determined point: {ray_slope_determined_point}, ray slope: {ray_slope}")

            # Step 1: Find the closest perpendicular bisector
            inter_slope, q_index, intersection, vertical_flag = self.find_closest_bisector(
                current_p, points, used_indices, ray_slope, ray_slope_determined_point
            )
            # print(f"Inter_slope: {inter_slope}, q_index: {q_index}, intersection: {intersection}, vertical_flag: {vertical_flag}")

            if intersection is None:
                special_flag = True
                break
            if iterations_number == 1 and q_index is None:
                initial_ray_slope_direction_flag = True
                # print(f"Can not find intersection point while (ray_slope, ray_slope_determined_point): {ray_slope, ray_slope_determined_point}")
                ray_slope_determined_point = (current_p_x + 1, current_p_y + 1)
                # print(f"Already change ray_slope_determined_point, after is {ray_slope, ray_slope_determined_point}")
                inter_slope, q_index, intersection, vertical_flag = self.find_closest_bisector(
                    current_p, points, used_indices, ray_slope, ray_slope_determined_point
                )
                # print(f"Inter_slope: {inter_slope}, q_index: {q_index}, intersection: {intersection}, vertical_flag: {vertical_flag}")

            # if (iterations_number == 1 and
            #     (intersection[0] < self.rectangleBoundingBox.x_range[0] or intersection[0] > self.rectangleBoundingBox.x_range[1] or intersection[1] < self.rectangleBoundingBox.y_range[0] or intersection[1] > self.rectangleBoundingBox.y_range[1])):
            #     intersection_x_new = (current_p_x + points[q_index][0]) / 2
            #     intersection_y_new = (current_p_y + points[q_index][1]) / 2
            #     intersection = (intersection_x_new, intersection_y_new)
            #     print(f"### New (Iteration 1) ### Inter_slope: {inter_slope}, q_index: {q_index}, intersection: {intersection}, vertical_flag: {vertical_flag}")

            # if intersection[1] < self.rectangleBoundingBox.y_range[0]:
            #     print("***Test")
            #     intersection_x_new = (current_p_x + points[q_index][0]) / 2
            #     intersection_y_new = (current_p_y + points[q_index][1]) / 2
            #     intersection = (intersection_x_new, intersection_y_new)
            #     print(f"### New Test ### Inter_slope: {inter_slope}, q_index: {q_index}, intersection: {intersection}, vertical_flag: {vertical_flag}")

            if iterations_number > 1 and preview_edge is not None and self.is_intersecting(current_p, intersection, preview_edge):
                if abs(intersection[0] - preview_edge[1][0]) < tol:
                    if intersection[0] < preview_edge[1][0]:
                        intersection_x_new = intersection[0] + 0.001
                    else:
                        intersection_x_new = intersection[0] - 0.001
                else:
                    if intersection[0] < preview_edge[0][0]:
                        intersection_x_new = intersection[0] + 0.001
                    else:
                        intersection_x_new = intersection[0] - 0.001
                intersection_y_new = inter_slope * (intersection_x_new - intersection[0]) + intersection[1]
                intersection = (intersection_x_new, intersection_y_new)
                # print(f"### New ### Inter_slope: {inter_slope}, q_index: {q_index}, intersection: {intersection}, vertical_flag: {vertical_flag}")

            if vertical_flag:
                # Step 2: Find the min and max ranges for y-coordinate
                y_min, y_max = float('-inf'), float('inf')
                y_0 = intersection[1]

                for r_index, r in enumerate(points):
                    if r_index == p_index or r_index == q_index:
                        continue  # Skip the current point and point q

                    # Calculate x-coordinate of intersection
                    y_r = self.calculate_intersection_y(current_p, r, intersection)

                    if y_r is None:
                        continue  # Skip parallel case (no intersection)

                    if y_r > y_0:
                        y_max = min(y_max, y_r)
                    elif y_r < y_0:
                        y_min = max(y_min, y_r)
                # print(f"y_min: {y_min}, y_max: {y_max}")

                if y_min == float('-inf') or y_max == float('inf'):
                    # print(f"### Need to cut ###")
                    if y_min == float('-inf') and y_max == float('inf'):
                        # print(f"### Need to cut two side ###")
                        collisions_number += 2
                    else:
                        collisions_number += 1
                    if y_min == float('-inf'):
                        y_min_final = self.rectangleBoundingBox.update_y_min()
                        y_max_final = y_max
                    elif y_max == float('inf'):
                        y_max_final = self.rectangleBoundingBox.update_y_max()
                        y_min_final = y_min
                    # print(f"After cutting y_min_final: {y_min_final}, y_max_final: {y_max_final}")
                else:
                    if y_min < self.rectangleBoundingBox.update_y_min():
                        # print(f"### Need to cut ###")
                        y_min = float('-inf')
                        y_min_final = self.rectangleBoundingBox.update_y_min()
                        collisions_number += 1
                    else:
                        # print(f"### No need to cut (y_min) ###")
                        y_min_final = y_min
                    if y_max > self.rectangleBoundingBox.update_y_max():
                        # print(f"### Need to cut ###")
                        y_max = float('inf')
                        y_max_final = self.rectangleBoundingBox.update_y_max()
                        collisions_number += 1
                    else:
                        # print(f"### No need to cut (y_max) ###")
                        y_max_final = y_max
                    # print(f"y_min final: {y_min_final},y_max_final: {y_max_final}")

                final_edge = self.calculate_final_edge_vertical_version(y_min_final, y_max_final, intersection)
                # print(f"Final edge: {final_edge}")

                if iterations_number == 1:
                    final_directed_edge = final_edge
                    initial_edge = final_edge
                    # print(f"Initial edge: {initial_edge}")
                else:
                    final_directed_edge = self.calculate_final_directed_edge(final_edge, ray_slope_determined_point, offset_flag)
                    # print(f"Final directed edge: {final_directed_edge}")

                if p_index < q_index:
                    self.edges.append(final_directed_edge)
                    # print(f"p_index < q_index, successfully added!")
                # else:
                    # print(f"p_index > q_index, not added!")

                if vis_steps:
                    self.plot_partial_result(current_p, final_directed_edge)
                used_indices.add(q_index)

                preview_edge = final_directed_edge

                if y_min == float('-inf') or y_max == float('inf'):
                    if collisions_number == 2:
                        # Finished Current Cell
                        # print("Finished Current Cell.\n")
                        break
                    else:
                        if (iterations_number == 1) and (initial_edge[0][0] == y_max or initial_edge[0][1] == y_min):
                            # print("Continue this side.\n")
                            result = self.point_offset_vertical_version(initial_edge[0], initial_edge[1])
                            ray_slope_determined_point = result[0]
                            offset_flag = result[1]
                        else:
                            # Continue Current Cell
                            # print("Switch to other side.\n")
                            preview_edge = initial_edge
                            result = self.point_offset_vertical_version(initial_edge[1], initial_edge[0])
                            ray_slope_determined_point = result[0]
                            offset_flag = result[1]
                else:
                    if math.isclose(final_directed_edge[0][0], initial_edge[1][0], rel_tol=1e-9) and math.isclose(final_directed_edge[0][1], initial_edge[1][1], rel_tol=1e-9):
                        # Finished Current Cell
                        # print("Finished Current Cell.\n")
                        break
                    else:
                        # Continue Current Cell
                        # print("Continue this side.\n")
                        result = self.point_offset_vertical_version(final_directed_edge[0], final_directed_edge[1])
                        ray_slope_determined_point = result[0]
                        offset_flag = result[1]
            else:
                # Step 2: Find the min and max ranges for x-coordinate
                x_min, x_max = float('-inf'), float('inf')
                x_0 = intersection[0]

                for r_index, r in enumerate(points):
                    if r_index == p_index or r_index == q_index:
                        continue  # Skip the current point and point q

                    x_r = self.calculate_intersection_x(current_p, r, intersection, inter_slope)

                    if x_r is None:
                        continue  # Skip parallel case (no intersection)

                    # if x_0 < self.rectangleBoundingBox.x_range[0]:
                    #     x_min = self.rectangleBoundingBox.x_range[0]
                    #     if x_r > x_min:
                    #         x_max = min(x_max, x_r)
                    # elif x_0 > self.rectangleBoundingBox.x_range[1]:
                    #     x_max = self.rectangleBoundingBox.x_range[1]
                    #     if x_r < x_max:
                    #         x_max = max(x_min, x_r)
                    # else:
                    #     if x_r > x_0:
                    #         x_max = min(x_max, x_r)
                    #     elif x_r < x_0:
                    #         x_min = max(x_min, x_r)

                    if x_r > x_0:
                        x_max = min(x_max, x_r)
                    elif x_r < x_0:
                        x_min = max(x_min, x_r)
                # print(f"x_min: {x_min}, x_max: {x_max}")

                if x_min == float('-inf') or x_max == float('inf'):
                    # print(f"### Need to cut ###")
                    if x_min == float('-inf') and x_max == float('inf'):
                        # print(f"### Need to cut two side ###")
                        collisions_number += 2
                    else:
                        collisions_number += 1
                    if x_min == float('-inf'):
                        x_min_final = self.rectangleBoundingBox.update_x_min(intersection, inter_slope)
                        x_max_final = x_max
                    elif x_max == float('inf'):
                        x_max_final = self.rectangleBoundingBox.update_x_max(intersection, inter_slope)
                        x_min_final = x_min
                    # print(f"After cutting x_min_final: {x_min_final}, x_max_final: {x_max_final}")
                else:
                    if x_min <= self.rectangleBoundingBox.update_x_min(intersection, inter_slope):
                        # print(f"### Need to cut ###")
                        x_min = float('-inf')
                        x_min_final = self.rectangleBoundingBox.update_x_min(intersection, inter_slope)
                        collisions_number += 1
                        collisions_number_change_flag = True
                    else:
                        # print(f"### No need to cut (x_min) ###")
                        x_min_final = x_min
                    if x_max >= self.rectangleBoundingBox.update_x_max(intersection, inter_slope):
                        # print(f"### Need to cut ###")
                        x_max = float('inf')
                        x_max_final = self.rectangleBoundingBox.update_x_max(intersection, inter_slope)
                        collisions_number += 1
                        collisions_number_change_flag = True
                    else:
                        # print(f"### No need to cut (x_max) ###")
                        x_max_final = x_max
                    # print(f"x_min_final: {x_min_final}, x_max_final: {x_max_final}")

                final_edge = self.calculate_final_edge_normal_version(x_min_final, x_max_final, inter_slope, intersection)
                # print(f"Final edge: {final_edge}")

                if ((math.isclose(final_edge[0][1], self.rectangleBoundingBox.y_range[0], rel_tol=1e-9) or
                     math.isclose(final_edge[0][1], self.rectangleBoundingBox.y_range[1], rel_tol=1e-9))
                        and not collisions_number_change_flag):
                    x_min = float('-inf')
                    x_max = float('inf')
                    collisions_number += 1

                if iterations_number == 1:
                    if initial_ray_slope_direction_flag:
                        final_directed_edge = final_edge
                    else:
                        final_directed_edge = (final_edge[1], final_edge[0])
                    initial_edge = final_directed_edge
                    # print(f"Initial edge: {initial_edge}")
                else:
                    final_directed_edge = self.calculate_final_directed_edge(final_edge, ray_slope_determined_point, offset_flag)
                    # print(f"Final directed edge: {final_directed_edge}")

                if x_min_final < x_max_final:
                    added_edge_number += 1

                if p_index < q_index and x_min_final < x_max_final:
                    self.edges.append(final_directed_edge)
                    # print(f"p_index < q_index and x_min_final < x_max_final, successfully added!")
                else:
                    # if p_index >= q_index:
                        # print(f"p_index >= q_index, not added!")
                    if x_min_final >= x_max_final:
                        collisions_number -= 1
                        # print(f"x_min_final >= x_max_final, not added!")

                if vis_steps:
                    self.plot_partial_result(current_p, final_directed_edge)
                used_indices.add(q_index)

                preview_edge = final_directed_edge

                if x_min == float('-inf') or x_max == float('inf'):
                    if collisions_number >= 2:
                        # Finished Current Cell
                        # print("Finished Current Cell.\n")
                        break
                    else:
                        if (iterations_number == 1) and (initial_edge[0][0] == x_max or initial_edge[0][1] == x_min):
                            # print("Continue this side.\n")
                            result = self.point_offset_normal_version(initial_edge[0], initial_edge[1], current_p)
                            ray_slope_determined_point = result[0]
                            offset_flag = result[1]
                        elif (iterations_number != 1) and (collisions_number == 1) and (added_edge_number == 2) and (initial_edge[0][0] < self.rectangleBoundingBox.x_range[0]):
                            # Continue Current Cell
                            # print("Continue this side.\n")
                            result = self.point_offset_normal_version(final_directed_edge[0], final_directed_edge[1], current_p)
                            ray_slope_determined_point = result[0]
                            offset_flag = result[1]
                        # elif special_flag:
                        #     # Continue Current Cell
                        #     print("Continue this side.\n")
                        #     result = self.point_offset_normal_version(final_directed_edge[0], final_directed_edge[1], current_p)
                        #     ray_slope_determined_point = result[0]
                        #     offset_flag = result[1]
                        else:
                            # Continue Current Cell
                            # print("Switch to other side.\n")
                            preview_edge = initial_edge
                            result = self.point_offset_normal_version(initial_edge[1], initial_edge[0], current_p)
                            ray_slope_determined_point = result[0]
                            offset_flag = result[1]
                else:
                    if math.isclose(final_directed_edge[0][0], initial_edge[1][0], rel_tol=1e-9) and math.isclose(final_directed_edge[0][1], initial_edge[1][1], rel_tol=1e-9):
                        # Finished Current Cell
                        # print("Finished Current Cell.\n")
                        break
                    else:
                        # Continue Current Cell
                        # print("Continue this side.\n")
                        result = self.point_offset_normal_version(final_directed_edge[0], final_directed_edge[1], current_p)
                        ray_slope_determined_point = result[0]
                        offset_flag = result[1]

            if intersection[1] < self.rectangleBoundingBox.y_range[0] or intersection[1] > self.rectangleBoundingBox.y_range[1]:
                # special_flag = True
                collisions_number -= 1
            ray_slope = self.calculate_next_ray_slope(current_p, ray_slope_determined_point)

        if special_flag:
            return True

    def find_closest_bisector(self, current_p, points, used_indices, ray_slope, ray_slope_determined_point):
        min_distance = float('inf')
        closest_q_index = None
        intersection = None
        intersection_slope = None
        final_vertical_flag = False

        for i, q in enumerate(points):
            if i in used_indices:
                continue  # Skip already used points

            result = self.calculate_intersection(current_p, q, ray_slope)
            if result is None:
                continue  # No valid intersection

            x_inter, y_inter, bisector_slope, vertical_flag = result

            rsdp_x, rsdp_y = ray_slope_determined_point
            current_p_x, current_p_y = current_p
            if ((x_inter < current_p_x) and (rsdp_x > current_p_x)) or ((x_inter > current_p_x) and (rsdp_x < current_p_x)):
                continue

            inter = (x_inter, y_inter)  # Intersection point as a tuple
            distance = self.calculate_distance(current_p, inter)

            if distance < min_distance:
                if vertical_flag:
                    min_distance = distance
                    closest_q_index = i
                    intersection = inter
                    intersection_slope = bisector_slope
                    final_vertical_flag = True
                else:
                    min_distance = distance
                    closest_q_index = i
                    intersection = inter
                    intersection_slope = bisector_slope
                    final_vertical_flag = False

        return intersection_slope, closest_q_index, intersection, final_vertical_flag

    def calculate_intersection(self, p, q, ray_slope):
        px, py = p
        qx, qy = q
        mx, my = (px + qx) / 2, (py + qy) / 2
        vertical_flag = False

        if (qy != py):
            # case: not vertical bisector
            if (ray_slope == (-(qx - px) / (qy - py))): # case: ray & bisector parallel
                return None
            if qx != px:
                bisector_slope = -(qx - px) / (qy - py)
                x_inter = (py - my - ray_slope * px + bisector_slope * mx) / (bisector_slope - ray_slope)
                y_inter = py + ray_slope * (x_inter - px)
                return x_inter, y_inter, bisector_slope, vertical_flag
            else:
                bisector_slope = -(qx - px) / (qy - py)
                y_inter = my
                x_inter = (y_inter - py) / ray_slope + px
                return x_inter, y_inter, bisector_slope, vertical_flag
        else:
            # case: vertical bisector
            vertical_flag = True
            x_inter = mx
            y_inter = py + ray_slope * (x_inter - px)
            return x_inter, y_inter, None, vertical_flag

    def calculate_distance(self, p, intersection):
        return math.sqrt((p[0] - intersection[0]) ** 2 + (p[1] - intersection[1]) ** 2)

    def calculate_intersection_y(self, p, r, intersection):
        px, py = p
        rx, ry = r
        ix, iy = intersection
        mx, my = (px + rx) / 2, (py + ry) / 2

        if (rx != px):
            if (ry == py):
                return None
            else:
                bisector_slope = -(rx - px) / (ry - py)
                y_inter = bisector_slope * (ix - mx) + my
                return y_inter
        else:
            y_inter = my
            return y_inter

    def calculate_intersection_x(self, p, r, intersection, inter_slope):
        px, py = p
        rx, ry = r
        ix, iy = intersection
        mx, my = (px + rx) / 2, (py + ry) / 2

        if (ry != py):
            if (inter_slope == (-(rx - px) / (ry - py))):
                return None
            if rx != px:
                bisector_slope = -(rx - px) / (ry - py)
                x_inter = (iy - my - inter_slope * ix + bisector_slope * mx) / (bisector_slope - inter_slope)
                return x_inter
            else:
                y_inter = my
                x_inter = (y_inter - iy) / inter_slope + ix
                return x_inter
        else:
            x_inter = mx
            return x_inter

    def calculate_final_edge_vertical_version(self, y_min, y_max, intersection):
        ix, iy = intersection
        return ((ix, y_min), (ix, y_max))

    def calculate_final_edge_normal_version(self, x_min, x_max, inter_slope, intersection):
        ix, iy = intersection
        y_min = inter_slope * (x_min - ix) + iy
        y_max = inter_slope * (x_max - ix) + iy
        return ((x_min, y_min), (x_max, y_max))

    def calculate_final_directed_edge(self, final_edge, ray_slope_determined_point, offset_flag):
        rsdp_x, rsdp_y = ray_slope_determined_point

        if offset_flag == 1:
            rsdp_y -= 0.0001
        elif offset_flag == 2:
            rsdp_x -= 0.0001
        elif offset_flag == 3:
            rsdp_x += 0.0001
        elif offset_flag == 4:
            rsdp_y -= 0.0001
        elif offset_flag == 5:
            rsdp_x += 0.0001
        elif offset_flag == 6:
            rsdp_y += 0.0001
        elif offset_flag == 7:
            rsdp_y += 0.0001
        elif offset_flag == 8:
            rsdp_x -= 0.0001

        ray_slope_determined_point_before = (rsdp_x, rsdp_y)

        if math.isclose(final_edge[0][0], ray_slope_determined_point_before[0], rel_tol=1e-9) and math.isclose(final_edge[0][1], ray_slope_determined_point_before[1], rel_tol=1e-9):
            return (final_edge[1], final_edge[0])
        else:
            return final_edge

    def point_offset_normal_version(self, current_start, current_end, current_p):
        start_x, start_y = current_start
        end_x, end_y = current_end
        p_x, p_y = current_p
        if end_x == start_x:
            new_current_start, offset_flag = self.point_offset_vertical_version(current_start, current_end)
            return new_current_start, offset_flag
        else:
            slope = (end_y - start_y) / (end_x - start_x)
        new_current_start = None
        offset_flag = -1

        ## line function: y = slope * (x - start_x) + start_y
        if p_y > slope * (p_x - start_x) + start_y:
            if slope < 0:
                if start_x < end_x:
                    list_current_start = list(current_start)
                    list_current_start[1] += 0.0001
                    new_current_start = tuple(list_current_start)
                    offset_flag = 1
                else:
                    list_current_start = list(current_start)
                    list_current_start[0] += 0.0001
                    new_current_start = tuple(list_current_start)
                    offset_flag = 2
            else:
                if start_x < end_x:
                    list_current_start = list(current_start)
                    list_current_start[0] -= 0.0001
                    new_current_start = tuple(list_current_start)
                    offset_flag = 3
                else:
                    list_current_start = list(current_start)
                    list_current_start[1] += 0.0001
                    new_current_start = tuple(list_current_start)
                    offset_flag = 4

        elif p_y < slope * (p_x - start_x) + start_y:
            if slope < 0:
                if start_x < end_x:
                    list_current_start = list(current_start)
                    list_current_start[0] -= 0.0001
                    new_current_start = tuple(list_current_start)
                    offset_flag = 5
                else:
                    list_current_start = list(current_start)
                    list_current_start[1] -= 0.0001
                    new_current_start = tuple(list_current_start)
                    offset_flag = 6
            else:
                if start_x < end_x:
                    list_current_start = list(current_start)
                    list_current_start[1] -= 0.0001
                    new_current_start = tuple(list_current_start)
                    offset_flag = 7
                else:
                    list_current_start = list(current_start)
                    list_current_start[0] += 0.0001
                    new_current_start = tuple(list_current_start)
                    offset_flag = 8
        return new_current_start, offset_flag

    def point_offset_vertical_version(self, current_start, current_end):
        start_x, start_y = current_start
        end_x, end_y = current_end
        new_current_start = None
        offset_flag = -1

        if start_y > end_y:
            list_current_start = list(current_start)
            list_current_start[1] += 0.0001
            new_current_start = tuple(list_current_start)
            offset_flag = 1
        else:
            list_current_start = list(current_start)
            list_current_start[1] -= 0.0001
            new_current_start = tuple(list_current_start)
            offset_flag = 6

        return new_current_start, offset_flag

    def calculate_next_ray_slope(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2

        if x1 == x2:
            return float('inf')

        slope = (y2 - y1) / (x2 - x1)
        return slope

    def is_intersecting(self, current_p, intersection, preview_edge):
        x0, y0 = current_p
        x1, y1 = intersection
        x2, y2 = preview_edge[0]
        x3, y3 = preview_edge[1]

        A = y1 - y0
        B = x0 - x1
        C = x1 * y0 - x0 * y1

        F1 = A * x2 + B * y2 + C
        F2 = A * x3 + B * y3 + C

        return F1 * F2 < 0