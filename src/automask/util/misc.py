def index_to_coords(index, grid_size):
    row, col = divmod(index, grid_size)
    return col / (grid_size - 1), row / (grid_size - 1)
