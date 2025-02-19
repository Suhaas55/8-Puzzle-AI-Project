from collections import deque

goal_state=('_','1','2','3','4','5','6','7','8') #for final end goal top left willl be empty

def dfs(initial_state, limit=30):
    stack = [(initial_state, [], 0)]  #Stack will include the state, path, and depth
    visited = set()

    while stack:
        state, path, depth = stack.pop()

        if depth > limit:  # Depth-limited search
            continue

        state_tuple = tuple(state)  # Storing tuplle version once
        visited.add(state_tuple)

        if state_tuple == goal_state:  #using the stored tuple
            return path  #returning the path to the goal state again

        for move, new_state in get_successors(state):  #Not needing for redundant depth check
            new_state_tuple = tuple(new_state)
            if new_state_tuple not in visited:
                stack.append((new_state, path + [move], depth + 1))  #To increase ther depth

    return None  # If no solutionwas found within the depth limit

# Just defining the possible moves globally ooverall to avoid redefining in every function call
MOVE_MAP = {      #for those index shifts
    'Left': -1,
    'Up': -3,
    'Right': 1,
    'Down': 3
}

def get_successors(state): #for like finding all of the possible movesfrom a certain state
    index = state.index('_')  # findign the empty tile position
    successors = []

    for move, idx_change in MOVE_MAP.items():
        new_index = index + idx_change

        # Checking if new index is within bounds
        if 0 <= new_index < 9:
            # To prevent moving left when on left edge & right when on right edge
            if (index % 3 == 0 and move == 'Left') or (index % 3 == 2 and move == 'Right'):
                continue  

            # this is for directly creating a new state without a separate swap function
            new_state = list(state)  
            new_state[index], new_state[new_index] = new_state[new_index], new_state[index]

            # Store move direction and the resulting new state
            successors.append((move, tuple(new_state)))  # Store as tuple for much better fficiency

    return successors

def read_input_file(file_path='input.txt'):  #basically just reads the puzzle state from input.txt
    with open(file_path, 'r') as file:
        return file.read().strip().split(',') or None
    
# Define move index shifts globally (so it doesn't get redefined every function call)
MOVE_INDICES = {'D': -3, 'U': 3, 'R': -1, 'L': 1}

def format_solution_path(state, path):
    formatted_path = []
    index = state.index('_')  # For finding the empty space position

    for move in path:
        tile_index = index + MOVE_INDICES[move]  # For finding tile to swap
        formatted_path.append(f"{state[tile_index]}{move}")  # Storing tile + move
        state[index], state[tile_index] = state[tile_index], state[index]  # Swappin
        index = tile_index  # Update empty space position

    return ''.join(formatted_path)

# For reading the initial state from file input.txt that we created(sepreately) add example state DONT FORGET LATER!
initial_state = read_input_file('input.txt')

# Execute DFS with depth limit
solution_path_dfs = dfs(initial_state, limit=90)

# Global mapping for move directions (to avoid redefining each time)
DIRECTION_MAPPING = {'Up': 'D', 'Down': 'U', 'Left': 'R', 'Right': 'L'}

# Basically if a solution is found then to format it
if solution_path_dfs:
    adjusted_solution_path = [DIRECTION_MAPPING[move] for move in solution_path_dfs]
    formatted_solution_dfs = format_solution_path(initial_state.copy(), adjusted_solution_path)
    formatted_solution_output_dfs = f"The solution of Q1.1.a is:\n{formatted_solution_dfs}"
else:
    formatted_solution_output_dfs = "No solution found within the depth limit"


print(formatted_solution_output_dfs) #DFS



