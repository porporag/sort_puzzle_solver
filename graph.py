from copy import deepcopy
class Board:
    def __init__(self, board, parent=None, from_which=-1, to_which=-1):
        """Initialize a board(state)

        Args:
            board [list[list]]: all bottles
            parent [Board]: the parent of current state
            from_which [int]: index of bottle which pops color blocks
            to_which [int]: index of bottle which push color blocks
        """
        self.board = board
        self.parent = parent
        self.from_which = from_which
        self.to_which = to_which

    def next_boards(self):
        """Successor function: all next states of the current state

        Returns: list[Board]
            All valid successive states from current state
        """
        next_states = []
        n_bottles = len(self.board)

        for i in range(n_bottles):
            if self.board[i].finished:
                continue
            pseudo_pop_res = self.board[i].pop(pseudo=True)
            if pseudo_pop_res is None:
                continue

            for j in range(n_bottles):
                if i == j or self.board[j].finished:
                    continue
                if self.board[j].push(pseudo_pop_res, pseudo=True) == -1:
                    continue

                # Heuristic pruning: avoid trivial moves
                if self.board[i].size == len(pseudo_pop_res) and self.board[j].size == 0:
                    continue

                # Minimize deep copy usage
                copied_board = self._copy_and_move(i, j)
                next_states.append(Board.board_factory(board=copied_board,
                                                       parent=self,
                                                       from_which=i,
                                                       to_which=j))

        return next_states

    def _copy_and_move(self, from_idx, to_idx):
        """Efficient deep copy and move function to avoid full deep copies"""
        # Only deep copy the affected bottles instead of the entire board
        new_board = list(self.board)  # Shallow copy of the board
        new_board[from_idx] = deepcopy(self.board[from_idx])
        new_board[to_idx] = deepcopy(self.board[to_idx])

        # Perform the actual move on the copied board
        new_board[to_idx].push(new_board[from_idx].pop())

        return new_board

    def state_checking(self):
        """Check if the current state is the goal state"""
        return all(self._consistent(bottle.data) for bottle in self.board)

    def _consistent(self, bottle_data):
        """Check if the bottle is in the desired goal state"""
        return len(bottle_data) == 4 and all(color == bottle_data[0] for color in bottle_data)

    @property
    def string(self):
        """String representation of the current state"""
        return "".join([bottle.string for bottle in self.board])

    @classmethod
    def board_factory(cls, board, parent=None, from_which=-1, to_which=-1):
        """Generate a new board (state)"""
        return cls(board, parent=parent, from_which=from_which, to_which=to_which)
