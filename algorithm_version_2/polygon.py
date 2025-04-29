class Polygon:
    def __init__(self, points):
        self.points = points
        self.edges = []
        self._build_edges()

    def _build_edges(self, close: bool = True) -> None:
        n = len(self.points)
        if n < 2:
            return

        for i in range(n - 1):
            self.edges.append((self.points[i], self.points[i + 1]))

        if close and n > 2:
            self.edges.append((self.points[-1], self.points[0]))