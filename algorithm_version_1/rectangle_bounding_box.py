class RectangleBoundingBox:
    def __init__(self, x_range, y_range):
        self.x_range = x_range
        self.y_range = y_range
        self.vertices = [(x_range[0], y_range[0]), (x_range[0], y_range[1]), (x_range[1], y_range[1]), (x_range[1], y_range[0])]

    def update_x_min(self, intersection, inter_slope):
        if inter_slope == 0:
            return self.x_range[0]
        ix, iy = intersection
        x_cut_1 = (self.y_range[0] - iy) / inter_slope + ix
        x_cut_2 = (self.y_range[1] - iy) / inter_slope + ix
        if inter_slope > 0:
            x_cut_1 = max(x_cut_1, self.x_range[0])
            return x_cut_1
        else:
            x_cut_2 = max(x_cut_2, self.x_range[0])
            return x_cut_2

    def update_x_max(self, intersection, inter_slope):
        if inter_slope == 0:
            return self.x_range[1]
        ix, iy = intersection
        x_cut_1 = (self.y_range[0] - iy) / inter_slope + ix
        x_cut_2 = (self.y_range[1] - iy) / inter_slope + ix
        if inter_slope > 0:
            x_cut_2 = min(x_cut_2, self.x_range[1])
            return x_cut_2
        else:
            x_cut_1 = min(x_cut_1, self.x_range[1])
            return x_cut_1

    def update_y_min(self):
        return self.y_range[0]

    def update_y_max(self):
        return self.y_range[1]

