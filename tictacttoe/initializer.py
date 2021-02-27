"""
The agent always plays o's, if its playing me I'll play x's.
This program declares the Board class and can detect wins, losses and draws given a board object.
x's are 1's and o's are 2'su
# TODO Link it to a GUI for manual play
"""
from numpy import base_repr
import copy
# There are 8 symmetries on the square. I use them because they keep numbers next to their neighbors.
symmetries = [[6, 3, 0, 7, 4, 1, 8, 5, 2],
              [8, 7, 6, 5, 4, 3, 2, 1, 0],
              [2, 5, 8, 1, 4, 7, 0, 3, 6],
              [6, 7, 8, 3, 4, 5, 0, 1, 2],
              [2, 1, 0, 5, 4, 3, 8, 7, 6],
              [0, 3, 6, 1, 4, 7, 2, 5, 8],
              [8, 5, 2, 7, 4, 1, 6, 3, 0],
              [0, 1, 2, 3, 4, 5, 6, 7, 8]]

# The three equivalence classes of wins
wins = [[1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1]]


class Board:
    global symmetries

    def __init__(self, number):
        self.number = number
        self.string = list(rebase(number).zfill(9))
        symmetrise = []
        for sym in symmetries:
            new_list = [0] * 9
            for j in range(9):
                new_list[sym[j]] = self.string[j]
            symmetrise.append(rebase(''.join([str(i) for i in new_list]), False))
        # symmetrise stores the entire equivalence class of this board.
        self.symmetrise = symmetrise

    def __eq__(self, other):
        for new_list in self.symmetrise:
            if new_list == other.number:
                return True
        return False

    def valuer(self, layout):
        # Checks if the given layout has player's x's winning
        for k in symmetries:
            now = Board(multi(k, layout)).string
            for j in wins:

                anded = [int(now[m] == '1' and j[m] == 1) for m in range(9)]
                # print(anded, j, layout)
                # print('now:',now)
                # print('j  :', j)
                if anded == j:
                    return True
        return False

    def win_lose(self):
        if self.valuer(self.string):
            return 'lose'
        # converts to check if o's have won
        elif self.valuer([str(int(m == '2')) for m in self.string]):
            return 'win'
        elif '0' not in self.string:
            return 'draw'
        else:
            return 'even'

    def actions(self, o=True):
        # x asks whether the computer is playing x's(True) or o's (False)
        ret_list = []
        for i in range(len(self.string)):
            if self.string[i] == '0':
                a = copy.deepcopy(self.string)
                a[i] = str(int(o) + 1)
                b = rebase(''.join(a), direction=False)
                ret_list.append(Board(b))
        return ret_list

    def __str__(self):
        line_1 = "|".join(self.between(self.string[0:3]))
        line_2 = "|".join(self.between(self.string[3:6]))
        line_3 = "|".join(self.between(self.string[6:9]))

        return "\n".join([line_1, line_2, line_3])

    def between(self, line):
        a = ''
        for i in line:
            if i=='0':
                a+=' '
            elif i=='1':
                a+='x'
            elif i=='2':
                a+='o'
        return a

    def __int__(self):
        return self.number


# Dictionary of board numbers to board preference indices, initially set to 0.5.
layouts = {}


def initializer():
    global layouts
    # Iterating through all possible layouts
    for board_init in range(3**9):
        board = Board(board_init)

        # Boards are equivalent to their transformations under the symmetries of a square. This allows me to store
        # fewer boards.
        # Check the 8 transformations.
        if abs(board.string.count('1')-board.string.count('2')) > 1:
            continue

        if check_in(board, layouts)[0]:
            continue

        # Preference index will always be 1 if win and 0 is draw or lose, so those aren't included in layouts.
        if board.win_lose()=='win':
            layouts[board.number]=1
            continue
        if board.win_lose() in ['lose', 'draw']:
            layouts[board.number]=0
            continue

        # Initialize all board preferences to 0.5, indexing them by their number 0<=n<19,683.
        layouts[board.number] = 0.5
    # How many I'm storing.
    print("Length of layouts is ", len(layouts))


def rebase(number, direction=True):
    # Converts base 10 to 3 if True and base 3 to 10 if False.
    if direction:
        return base_repr(number, base=3).zfill(9)
    else:
        counter = 0
        # print(number)
        for i in range(len(number)):
            counter += int(number[i])*3**(len(number)-i-1)
    return counter


def check_in(board, dictionary):
    # Checks if the equivalence class of a board is included in the dictionary already.
    for hash_1 in board.symmetrise:
        # print(''.join(hash_1))
        if hash_1 in dictionary:
            return [True, hash_1]
    return [False]


def multi(sym, board):
    # board is list and sym is just a list
    new_board = ['0'] * 9
    # print(board.number)
    # print(board, len(board))
    for k in range(9):
        new_board[k] = board[sym[k]]
    # Returns int(new_board)
    return rebase(''.join(new_board), False)
