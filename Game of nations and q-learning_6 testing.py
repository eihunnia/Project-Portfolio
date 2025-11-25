import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
#Add age parameter to members, to speed up simulator  if cell is 5 
# Constants
GRID_SIZE = 300
NATIONS = [1, 2, 3, 4]
EMPTY, RESOURCE, LION = 0, 5, 6
ACTIONS = ["expand", "attack", "move_to_resource", "lion_attack"]

# Q-learning Parameters
alpha = 0.1   # Learning rate
gamma = 0.7   # Discount factor for future rewards
epsilon = 0.1 # Exploration rate
Q_table = {}

# Logging Data
log_data = {
    "step": [],
    "nation_1_cells": [],
    "nation_2_cells": [],
    "nation_3_cells": [],
    "nation_4_cells": [],
    "total_rewards_1": [],
    "total_rewards_2": [],
    "total_rewards_3": [],
    "total_rewards_4": [],
    "resource_count": [],
    "lion_count": [],
}

# Color Mapping
COLOR_MAP = {
    EMPTY: [0, 0, 0],       # Black (Empty)
    1: [1, 0, 0],           # Red (Nation 1)
    2: [0, 0, 1],           # Blue (Nation 2)
    3: [1, 1, 0],           # Yellow (Nation 3)
    4: [0.5, 0, 0.5],       # Purple (Nation 4)
    RESOURCE: [0, 1, 0],    # Green (Resources)
    LION: [0.96, 0.63, 0.07] # Orange (Lions)
}

# Initialize grid
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
total_rewards = {nation: 0 for nation in NATIONS}

def initialize_grid():
    """Randomly place nations, resources, and lions on the grid."""
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rnd = random.random()
            if rnd < 0.05:
                grid[i, j] = random.choice(NATIONS)
            elif rnd < 0.07:
                grid[i, j] = RESOURCE
            elif rnd < 0.08:
                grid[i, j] = LION

def get_neighbours(x, y):
    """Return a list of valid neighbor coordinates."""
    neighbours = [(x+dx, y+dy) for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]]
    return [(nx, ny) for nx, ny in neighbours if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE]

def get_state(x, y):
    """Get a 3x3 neighborhood state as a tuple."""
    return tuple(grid[max(0, x-1):min(GRID_SIZE, x+2), max(0, y-1):min(GRID_SIZE, y+2)].flatten())

def choose_action(state):
    """Epsilon-greedy action selection."""
    if state not in Q_table:
        Q_table[state] = {a: 0 for a in ACTIONS}
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    return max(Q_table[state], key=Q_table[state].get)

def q_learning_update(state, action, reward, new_state):
    """Q-learning update step."""
    if new_state not in Q_table:
        Q_table[new_state] = {a: 0 for a in ACTIONS}
    max_q_new = max(Q_table[new_state].values())
    Q_table[state][action] += alpha * (reward + gamma * max_q_new - Q_table[state][action])

def update_grid():
    """Perform one step of simulation including Q-learning updates."""
    new_grid = grid.copy()

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if grid[i, j] in NATIONS:
                nation = grid[i, j]
                state = get_state(i, j)
                action = choose_action(state)
                reward = 0     

                # Get neighboring info
                neighbours = get_neighbours(i, j)
                nation_neighbours = [grid[nx, ny] for nx, ny in neighbours if grid[nx, ny] in NATIONS]
                lion_neighbours = [(nx, ny) for nx, ny in neighbours if grid[nx, ny] == LION]
                empty_cells = [(nx, ny) for nx, ny in neighbours if grid[nx, ny] == EMPTY]            #ALTERNATE NAME IS EMPTY_NEIGHBOURS
                resource_neighbours = [(nx, ny) for nx, ny in neighbours if grid[nx, ny] == RESOURCE]
                enemy_neighbours = [(nx, ny) for nx, ny in neighbours if grid[nx, ny] in NATIONS and grid[nx, ny] != nation]

                # Lion Attack: If a nation is adjacent to a lion, the nation cell dies
                if lion_neighbours:
                    action = "lion_attack"
                    new_grid[i, j] = EMPTY
                    reward -= 10  # Penalty for being killed by a lion

                elif action == "expand" and empty_cells:
                    new_x, new_y = random.choice(empty_cells)
                    new_grid[new_x, new_y] = nation
                    reward += 10  # Reward for expansion

                elif action == "attack" and enemy_neighbours:
                    new_x, new_y = random.choice(enemy_neighbours)
                    new_grid[new_x, new_y] = nation
                    reward += 15  # Reward for successful attack

                elif action == "move_to_resource" and resource_neighbours:
                    new_x, new_y = random.choice(resource_neighbours)
                    new_grid[new_x, new_y] = nation
                    reward += 20  # Reward for collecting resource

                # Survival & Death Rules
                if len(nation_neighbours) < 2 or len(nation_neighbours) > 4:
                    new_grid[i, j] = EMPTY  # Cell dies
                    reward -= 5  # Penalty for dying

                # Birth Rule
                if grid[i, j] == EMPTY and len(nation_neighbours) == 3:
                    majority_nation = max(set(nation_neighbours), key=nation_neighbours.count)
                    new_grid[i, j] = majority_nation
                    reward += 8  # Reward for successful birth

                # Q-learning update
                new_state = get_state(i, j)
                q_learning_update(state, action, reward, new_state)
                total_rewards[nation] += reward

    return new_grid

def log_simulation(step):
    """Log the current simulation state."""
    nation_counts = {nation: np.sum(grid == nation) for nation in NATIONS}
    resource_count = np.sum(grid == RESOURCE)
    lion_count = np.sum(grid == LION)

    log_data["step"].append(step)
    for nation in NATIONS:
        log_data[f"nation_{nation}_cells"].append(nation_counts[nation])
        log_data[f"total_rewards_{nation}"].append(total_rewards[nation])

    log_data["resource_count"].append(resource_count)
    log_data["lion_count"].append(lion_count)

def plot_simulation():
    """Plot the nation expansion and rewards over time with correct colors."""
    df = pd.DataFrame(log_data)

    # Define the correct colors for nations
    nation_colors = {
        1: "red",
        2: "blue",
        3: "yellow",
        4: "purple",
    }

    plt.figure(figsize=(12, 6))

    # Plot Nation Expansion Over Time
    plt.subplot(1, 2, 1)
    for nation in NATIONS:
        plt.plot(df["step"], df[f"nation_{nation}_cells"], label=f"Nation {nation}", color=nation_colors[nation])
    plt.xlabel("Step")
    plt.ylabel("Cells Owned")
    plt.title("Nation Expansion Over Time")
    plt.legend()

    # Plot Total Rewards Over Time
    plt.subplot(1, 2, 2)
    for nation in NATIONS:
        plt.plot(df["step"], df[f"total_rewards_{nation}"], label=f"Nation {nation}", color=nation_colors[nation])
    plt.xlabel("Step")
    plt.ylabel("Total Rewards")
    plt.title("Nation Rewards Over Time")
    plt.legend()

    plt.show()

def render_grid():
    """Render the grid."""
    rgb_grid = np.array([[COLOR_MAP[cell] for cell in row] for row in grid])
    plt.imshow(rgb_grid)
    plt.axis("off")

# Simulation Loop
initialize_grid()
for step in range(100):
    grid = update_grid()
    log_simulation(step)
    render_grid()
    plt.pause(0.1)

plot_simulation()
plt.show()