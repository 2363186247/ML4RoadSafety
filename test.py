from ml_for_road_safety import TrafficAccidentDataset

# Creating the dataset as PyTorch Geometric dataset object
# 使用正确的数据路径
dataset = TrafficAccidentDataset(state_name="MA", data_dir="./ML4RoadSafety/ml_for_road_safety/data")

# Loading the accident records and traffic network features of a particular month
data = dataset.load_monthly_data(year = 2022, month = 1)

# Pytorch Tensors storing the list of edges with accidents and accident numbers
accidents, accident_counts = data["accidents"], data["accident_counts"]

# Pytorch Tensors of node features, edge list, and edge features
x, edge_index, edge_attr = data["x"], data["edge_index"], data["edge_attr"]

print(data)