from algorithm_version_2.constant_workspace_algorithm_update import ConstantWorkspaceAlgorithm_version_2
from algorithm_version_2.polygon import Polygon
import tracemalloc
import time
import csv

n = 500
seed = 2

# Define a set of points
points = []

base_path = "/Users/evan/Desktop/Capstone_Project/Implementation/constant_workspace_algorithm/evan_file/"

with open(f'{base_path}INPUT/{n}points_seeds.csv', 'r', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['seed']) == seed:
            x = float(row['x'])
            y = float(row['y'])
            points.append((x, y))


polygon = Polygon([
    (-1000000, -1000000),
    (-1000000, 1000000),
    (1000000, 1000000),
    (1000000, -1000000)
])

# Initialize the algorithm_version_2
v = ConstantWorkspaceAlgorithm_version_2(polygon)

start_time = time.perf_counter()

# Create the diagram
v.create_diagram(points=points, vis_steps=False, vis_result=True)

end_time = time.perf_counter()

print(f"time: {end_time - start_time}")
