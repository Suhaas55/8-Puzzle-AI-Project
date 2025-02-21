from collections import deque
import heapq
from math import sqrt

goal_state = ('1','2','3','8','_','4','7','6','5') #for spiral pattern with empty center in middle

#DFS
def dfs(initial_state, limit=30):
    stack = [(initial_state, [], 0)]  #Stack will include the state, path, and depth
    visited = set()

    while stack:
        state, path, depth = stack.pop()

        if depth > limit:  #depth limited searhc
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
    
# Defining move index shifts globally (so it doesn't get redefined every function call)
MOVE_INDICES = {'D': -3, 'U': 3, 'R': -1, 'L': 1}

def format_solution_path(state, path):
    formatted_path = []
    index = state.index('_')  # For finding the empty space position

    for move in path:
        tile_index = index + MOVE_INDICES[move]  # For finding tile to swap
        formatted_path.append(f"{state[tile_index]}{move}")  # Storing tile + move
        state[index], state[tile_index] = state[tile_index], state[index]  # Swappin
        index = tile_index  # Update empty space position

    return ','.join(formatted_path)

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
    formatted_solution_output_dfs = f"The solution of Q2.1.a is:\n{formatted_solution_dfs}"
else:
    formatted_solution_output_dfs = "No solution found within the depth limit"


#BFS
def bfs(initial_state):
    queue = deque([(initial_state, [])])  #BFS uses a queue for the FIFO order
    visited = set()

    while queue:
        state, path = queue.popleft()
        state_tuple = tuple(state)  # To store tuple version once
        visited.add(state_tuple)

        if state_tuple == goal_state:  #comparin with goal state
            return path  # after finding the solution, return path

        for move, new_state in get_successors(state):  
            new_state_tuple = tuple(new_state)
            if new_state_tuple not in visited:
                queue.append((new_state, path + [move]))  # Add to queue

    return None  # No solution found

# Reading initial state from file
initial_state = read_input_file('input.txt')

# Execute BFS
solution_path_bfs = bfs(initial_state)

# If BFS finds a solution then format it
if solution_path_bfs:
    adjusted_solution_path_bfs = [DIRECTION_MAPPING[move] for move in solution_path_bfs]  #Convert moves
    formatted_solution_bfs = format_solution_path(initial_state.copy(), adjusted_solution_path_bfs)  #Format path
    formatted_solution_output_bfs = f"The solution of Q2.1.b is:\n{formatted_solution_bfs}"
else:
    formatted_solution_output_bfs = "No solution found with BFS"

#This is for performing Uniform Cost Search
def ucs(initial_state):
    frontier = []  #the priority queue
    heapq.heappush(frontier, (0, initial_state, []))  #(cost, state, path)
    visited_cost = {}  # Dictionary to track the lowest cost to reach each state

    while frontier:
        cost, state, path = heapq.heappop(frontier)
        state_tuple = tuple(state)  #To store tuple version once

        # If this state has already been found at a lower costthen skipping it
        if state_tuple in visited_cost and visited_cost[state_tuple] <= cost:
            continue
        visited_cost[state_tuple] = cost

        if state_tuple == goal_state:  # Checking if the goal that was stated is reached
            return path  

        for move, new_state in get_successors(state):
            new_state_tuple = tuple(new_state)
            new_cost = cost + 1  # Uniform cost, every move costs 1
            if new_state_tuple not in visited_cost or new_cost < visited_cost[new_state_tuple]:
                heapq.heappush(frontier, (new_cost, new_state, path + [move]))

    return None  #If no solution found

# For executing Uniform Cost Search
solution_path_ucs = ucs(initial_state)

# If UCS finds a solution then format it
if solution_path_ucs:
    adjusted_solution_path_ucs = [DIRECTION_MAPPING[move] for move in solution_path_ucs]
    formatted_solution_ucs = format_solution_path(initial_state.copy(), adjusted_solution_path_ucs)
    formatted_solution_output_ucs = f"The solution of Q2.1.c is:\n{formatted_solution_ucs}"
else:
    formatted_solution_output_ucs = "No solution found with UCS"

#For calculating the Manhattan distance heuristic for a state
def manhattan_distance(state):
    # Modified for spiral goal pattern fo rthis question so ADDED
    index_map = {val: idx for idx, val in enumerate(goal_state)}  #to basically precompute goal positions
    distance = 0
    for idx, num in enumerate(state):
        if num != '_':  # Skipping any empty tile
            goal_idx = index_map[num]  #for trying to get correct position from goal state
            distance += abs(idx % 3 - goal_idx % 3) + abs(idx // 3 - goal_idx // 3)  #The Manhattan distance formula

    return distance

# A* search start
def a_star_search(initial_state):
    frontier = []
    heapq.heappush(frontier, (0, initial_state, []))  # (cost + heuristic, state, path)
    cost_so_far = {tuple(initial_state): 0}  # Cost from start to the node

    while frontier:
        _, current_state, path = heapq.heappop(frontier)
        current_tuple = tuple(current_state)  #To store tuple version once

        if current_tuple == goal_state:  # Checking if goal is reached
            return path  

        for move, new_state in get_successors(current_state):
            new_state_tuple = tuple(new_state)
            new_cost = cost_so_far[current_tuple] + 1  # Each move costs 1

            if new_state_tuple not in cost_so_far or new_cost < cost_so_far[new_state_tuple]:
                cost_so_far[new_state_tuple] = new_cost
                priority = new_cost + manhattan_distance(new_state)
                heapq.heappush(frontier, (priority, new_state, path + [move]))

    return None

# Execute A* search
solution_path_a_star = a_star_search(initial_state)

# If A* finds a solution then to format it
if solution_path_a_star:
    adjusted_solution_path_a_star = [DIRECTION_MAPPING[move] for move in solution_path_a_star]  # Convert hte moves
    formatted_solution_a_star = format_solution_path(initial_state.copy(), adjusted_solution_path_a_star)  # Formatign the path
    formatted_solution_output_a_star = f"The solution of Q2.1.d is:\n{formatted_solution_a_star}"
else:
    formatted_solution_output_a_star = "No solution found with A*"


# Function to calculate the Euclidean distance heuristic for a state that is a straight line
def straight_line_distance(state):
    index_map = {val: idx for idx, val in enumerate(goal_state)}  #Trtying to prerecompute goal positions
    distance = 0

    for idx, num in enumerate(state):
        if num != '_':  #Skipping empty tile
            goal_idx = index_map[num]  #For getting correct position from goal state
            x1, y1 = idx % 3, idx // 3
            x2, y2 = goal_idx % 3, goal_idx // 3
            distance += sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  #Euclidean distance formula

    return distance

# Function to perform A* search with Euclidean Distance heuristic that is a straight line.
def a_star_search_sld(initial_state):
    frontier = []
    heapq.heappush(frontier, (0, initial_state, []))  # (cost + heuristic, state, path)
    cost_so_far = {tuple(initial_state): 0}  #Cost from start to node

    while frontier:
        _, current_state, path = heapq.heappop(frontier)
        current_tuple = tuple(current_state)  #To store tuple version once

        if current_tuple == goal_state:  #Goal test
            return path  

        for move, new_state in get_successors(current_state):
            new_state_tuple = tuple(new_state)
            new_cost = cost_so_far[current_tuple] + 1  #Assuming that each move costs 1

            if new_state_tuple not in cost_so_far or new_cost < cost_so_far[new_state_tuple]:
                cost_so_far[new_state_tuple] = new_cost
                priority = new_cost + straight_line_distance(new_state)
                heapq.heappush(frontier, (priority, new_state, path + [move]))

    return None

#To execute A* search with Euclidean Distance heuristic
solution_path_a_star_sld = a_star_search_sld(initial_state)

# If A* finds a solution then formatting it
if solution_path_a_star_sld:
    adjusted_solution_path_a_star_sld = [DIRECTION_MAPPING[move] for move in solution_path_a_star_sld]  # Convert moves
    formatted_solution_a_star_sld = format_solution_path(initial_state.copy(), adjusted_solution_path_a_star_sld)  # Format path
    formatted_solution_output_a_star_sld = f"The solution of Q2.1.e is:\n{formatted_solution_a_star_sld}"
else:
    formatted_solution_output_a_star_sld = "No solution found with A* using thee Straight Line Distance"


print(formatted_solution_output_dfs) #DFS
print(formatted_solution_output_bfs) #BFS
print(formatted_solution_output_ucs) #UCS
print(formatted_solution_output_a_star) #ASS
print(formatted_solution_output_a_star_sld) #ASSLD

#for Q2.2
'''
from collections import deque
import heapq
from math import sqrt

goal_state = ('1','2','3','8','_','4','7','6','5') #for spiral pattern with empty center in middle

#DFS
def dfs(initial_state, limit=30):
    stack = [(initial_state, [], 0)]  #Stack will include the state, path, and depth
    visited = set()
    node_expansions = 0  # Track number of expansions

    while stack:
        state, path, depth = stack.pop()

        if depth > limit:  #depth limited search
            continue

        state_tuple = tuple(state)  # Storing tuple version once
        visited.add(state_tuple)

        if state_tuple == goal_state:  
            return path, node_expansions  

        node_expansions += 1  # Counting expansions
        for move, new_state in get_successors(state):  
            new_state_tuple = tuple(new_state)
            if new_state_tuple not in visited:
                stack.append((new_state, path + [move], depth + 1))  

    return None, node_expansions  

# Just defining the possible moves globally overall to avoid redefining in every function call
MOVE_MAP = {      #for those index shifts
    'Left': -1,
    'Up': -3,
    'Right': 1,
    'Down': 3
}

def get_successors(state): #for like finding all of the possible moves from a certain state
    index = state.index('_')  # finding the empty tile position
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
            successors.append((move, tuple(new_state)))  # Store as tuple for much better efficiency

    return successors


#BFS
def bfs(initial_state):
    queue = deque([(initial_state, [])])  
    visited = set()
    node_expansions = 0  

    while queue:
        state, path = queue.popleft()
        state_tuple = tuple(state)  
        visited.add(state_tuple)

        if state_tuple == goal_state:  
            return path, node_expansions  

        node_expansions += 1  
        for move, new_state in get_successors(state):  
            new_state_tuple = tuple(new_state)
            if new_state_tuple not in visited:
                queue.append((new_state, path + [move]))  

    return None, node_expansions  

#UCS
def ucs(initial_state):
    frontier = []  
    heapq.heappush(frontier, (0, initial_state, []))  
    visited_cost = {}  
    node_expansions = 0  

    while frontier:
        cost, state, path = heapq.heappop(frontier)
        state_tuple = tuple(state)  

        if state_tuple in visited_cost and visited_cost[state_tuple] <= cost:
            continue
        visited_cost[state_tuple] = cost

        if state_tuple == goal_state:  
            return path, node_expansions  

        node_expansions += 1  
        for move, new_state in get_successors(state):
            new_state_tuple = tuple(new_state)
            new_cost = cost + 1  
            if new_state_tuple not in visited_cost or new_cost < visited_cost[new_state_tuple]:
                heapq.heappush(frontier, (new_cost, new_state, path + [move]))

    return None, node_expansions  

# For calculating the Manhattan distance heuristic for a state
def manhattan_distance(state):
    # Modified for spiral goal pattern for this question so ADDED
    index_map = {val: idx for idx, val in enumerate(goal_state)}  #to basically precompute goal positions
    distance = 0
    for idx, num in enumerate(state):
        if num != '_':  # Skipping any empty tile
            goal_idx = index_map[num]  #for trying to get correct position from goal state
            distance += abs(idx % 3 - goal_idx % 3) + abs(idx // 3 - goal_idx // 3)  #The Manhattan distance formula

    return distance


#A* Search with Manhattan Distance
def a_star_search(initial_state):
    frontier = []
    heapq.heappush(frontier, (0, initial_state, []))  
    cost_so_far = {tuple(initial_state): 0}  
    node_expansions = 0  

    while frontier:
        _, current_state, path = heapq.heappop(frontier)
        current_tuple = tuple(current_state)  

        if current_tuple == goal_state:  
            return path, node_expansions  

        node_expansions += 1  
        for move, new_state in get_successors(current_state):
            new_state_tuple = tuple(new_state)
            new_cost = cost_so_far[current_tuple] + 1  
            if new_state_tuple not in cost_so_far or new_cost < cost_so_far[new_state_tuple]:
                cost_so_far[new_state_tuple] = new_cost
                priority = new_cost + manhattan_distance(new_state)
                heapq.heappush(frontier, (priority, new_state, path + [move]))

    return None, node_expansions  

# Function to calculate the Euclidean distance heuristic for a state that is a straight line
def straight_line_distance(state):
    index_map = {val: idx for idx, val in enumerate(goal_state)}  #Trying to precompute goal positions
    distance = 0

    for idx, num in enumerate(state):
        if num != '_':  #Skipping empty tile
            goal_idx = index_map[num]  #For getting correct position from goal state
            x1, y1 = idx % 3, idx // 3
            x2, y2 = goal_idx % 3, goal_idx // 3
            distance += sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  #Euclidean distance formula

    return distance


#A* Search with Euclidean Distance
def a_star_search_sld(initial_state):
    frontier = []
    heapq.heappush(frontier, (0, initial_state, []))  
    cost_so_far = {tuple(initial_state): 0}  
    node_expansions = 0  

    while frontier:
        _, current_state, path = heapq.heappop(frontier)
        current_tuple = tuple(current_state)  

        if current_tuple == goal_state:  
            return path, node_expansions  

        node_expansions += 1  
        for move, new_state in get_successors(current_state):
            new_state_tuple = tuple(new_state)
            new_cost = cost_so_far[current_tuple] + 1  
            if new_state_tuple not in cost_so_far or new_cost < cost_so_far[new_state_tuple]:
                cost_so_far[new_state_tuple] = new_cost
                priority = new_cost + straight_line_distance(new_state)
                heapq.heappush(frontier, (priority, new_state, path + [move]))

    return None, node_expansions  

# fpr running all searches and ofr print node expansions
start_state = ('2', '8', '1', '_', '5', '3', '6', '7', '4')

dfs_result, dfs_expansions = dfs(start_state)
bfs_result, bfs_expansions = bfs(start_state)
ucs_result, ucs_expansions = ucs(start_state)
a_star_m_result, a_star_m_expansions = a_star_search(start_state)
a_star_sld_result, a_star_sld_expansions = a_star_search_sld(start_state)

#output
print("DFS Node Expansions:", dfs_expansions)
print("BFS Node Expansions:", bfs_expansions)
print("UCS Node Expansions:", ucs_expansions)
print("A* Manhattan Node Expansions:", a_star_m_expansions)
print("A* Euclidean Node Expansions:", a_star_sld_expansions)
'''