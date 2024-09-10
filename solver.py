#%%
import copy


def is_winning(state,max_length):
    for i in range(len(state)):
        if len(state[i]) > 0 and len(state[i]) != max_length:
            return False
        elif len(state[i]) == max_length:
            if not all(state[i][0] == color_index for color_index in state[i]):
                return False
    return True

def generate_neighbors(state,max_length):
    candidate_rows = []
    for i in range(len(state)):
        if len(state[i]) < max_length:
            if len(state[i]) == 0:
                candidate_rows.append((i, -1))
            else:
                candidate_rows.append((i, state[i][-1]))

    for i in range(len(state)):
        if len(state[i]) > 0:
            color_index = state[i][-1]
            for j, dst_color_index in candidate_rows:
            
                if j != i and (color_index == dst_color_index or dst_color_index == -1):
                    new_state = [
                        [k for k in row] for row in state
                    ]
                    new_state[i].pop()
                    new_state[j].append(color_index)

                    new_state = tuple(tuple(row) for row in new_state)
                    yield new_state, f"move from {i} to {j}"



'''Implementation of depth first search algorithm'''

def dfs(board,max_length):

    frontier = [[board, []]]
    seen_states = set()
    seen_states.add(tuple(sorted(board)))
    
    i = 0
    while True:
        if len(frontier) == 0:
            print("No solution found")
            break
        state, history = frontier.pop()
        if is_winning(state,max_length):
            print(f"Solution of size {len(history)} found in {i} iterations")
            break
        else:
            for neighbor, text_description in generate_neighbors(state,max_length):
                sorted_neighbor = tuple(sorted(neighbor))
                if sorted_neighbor not in seen_states:
                    new_history = copy.deepcopy(history) + [text_description]
                    frontier.append([neighbor, new_history])
                    seen_states.add(sorted_neighbor)
        i += 1
    return history