import numpy as np
import matplotlib.pyplot as plt

# parameters
grid_size = 100
init_cancer_cells = 1000
init_antioxidants = 500
iterations = 1000
antioxidant_effect = 0.05

# initialize grid with random cancer cells and antioxidants
grid = np.zeros((grid_size, grid_size))
for _ in range(init_cancer_cells):
    x, y = np.random.randint(0, grid_size, 2)
    grid[x, y] = 1
    # cancer cell
for _ in range(init_antioxidants):
    x, y = np.random.randint(0, grid_size, 2)
    grid[x, y] = 2

def count_neighbors(x, y, value):
    # count neighbouring cells with given value
    count = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = (x + dx) % grid_size , (y + dy) % grid_size
            if grid[nx, ny] == value:
                count += 1

    return count


def update_grid():
    # update grid for one time step
    global grid
    new_grid = grid.copy()
    for x in range(grid_size):
        for y in range(grid_size):
            if grid[x, y] == 1:
                antioxidant_nearby = count_neighbors(x, y, 2)
                if np.random.rand()< antioxidant_effect * antioxidant_nearby:
                    new_grid[x, y] = 0
            elif grid[x, y] == 2:
                pass



# simulation loop
for _ in range(iterations):
    update_grid()


# visualization
plt.imshow(grid, cmap = "tab10", vmin = 0, vmax= 2)
plt.title("Antioxidants and cancer cells after {} iterations".format(iterations))
# plt.show()