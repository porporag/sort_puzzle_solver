from graph import Bottle


def get_start_board():
    """Start board of the game level n

    Modify this function if you want to play new game level

    NOTE: COLORS SHOULD BE INPUTED FROM BOTTOL BOTTOM TO TOP
    """
    N_BOTTLES = 14  # total number of bottles including empty bottles
    global_board = []
    bottles = [[] for _ in range(N_BOTTLES)]
    

    for i in range(N_BOTTLES):
        _bottle = Bottle(i, bottles[i])
        global_board.append(_bottle)

    return global_board