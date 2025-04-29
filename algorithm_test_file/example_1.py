from algorithm_version_2.constant_workspace_algorithm_update import ConstantWorkspaceAlgorithm_version_2
from algorithm_version_2.polygon import Polygon

# Define a set of points
points = [
    (1, 1),
    (20, 1),
    (15, 20),
    (13, 10),
    (12, 5),
    (5, 12),
    (-5, 20),
    (19, 16),
    (10, 12),
    (9, 10)
]

polygon = Polygon([
    (-10, -10),
    (-10, 30),
    (30, 30),
    (30, -10)
])

# Initialize the algorithm_version_2
v = ConstantWorkspaceAlgorithm_version_2(polygon)

# Create the diagram
v.create_diagram(points=points, vis_steps=False, vis_result=True)
