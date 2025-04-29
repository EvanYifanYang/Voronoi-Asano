from algorithm_version_2.constant_workspace_algorithm_update import ConstantWorkspaceAlgorithm_version_2
from algorithm_version_2.polygon import Polygon

# Define a set of points
points = [
    (2.5, 2.5), (4, 7.5), (7.5, 2.5), (6, 7.5), (4, 4), (3, 3), (6, 3)
]

polygon = Polygon([
    (2.5, 10), (5, 10), (10, 5), (10, 2.5), (5, 0), (2.5, 0), (0, 2.5), (0, 5)
])

# Initialize the algorithm_version_2
v = ConstantWorkspaceAlgorithm_version_2(polygon)

# Create the diagram
v.create_diagram(points=points, vis_steps=False, vis_result=True)
