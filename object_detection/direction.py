from enum import Enum


class Direct(Enum):
    STRAIGHT = 0
    LEFT = 1
    RIGHT = 2
    SLIGHTLY_LEFT = 3
    SLIGHTLY_RIGHT = 4
    U_TURN = 5
    ARRIVED = 6
