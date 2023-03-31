class GraphData:
    def __init__(self, grid_name, train_data, val_data, test_data, edge_index, edge_attr):
        self.grid_name = grid_name
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.edge_index = edge_index
        self.edge_attr = edge_attr
    """
    @property
    def train_data(self):
        return self.train_data

    @train_data.setter
    def train_data(self, lst):
        self.train_data = lst

    @property
    def val_data(self):
        return self.val_data

    @val_data.setter
    def val_data(self, lst):
        self.val_data = lst

    @property
    def test_data(self):
        return self.test_data

    @test_data.setter
    def test_data(self, lst):
        self.test_data = lst

    @property
    def edge_index(self):
        return self.edge_index

    @edge_index.setter
    def edge_index(self, tensor):
        self.edge_index = tensor

    @property
    def edge_attr(self):
        return self.edge_attr

    @edge_attr.setter
    def edge_attr(self, tensor):
        self.edge_attr = tensor

    """

