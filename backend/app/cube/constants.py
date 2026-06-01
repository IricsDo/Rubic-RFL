FACE_ORDER = ("U", "R", "F", "D", "L", "B")
FACE_TO_AXIS = {
    "U": ("y", 1),
    "D": ("y", -1),
    "R": ("x", 1),
    "L": ("x", -1),
    "F": ("z", 1),
    "B": ("z", -1),
}
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
VALID_SUFFIXES = ("", "'", "2")
ALL_MOVES = tuple(
    face + suffix for face in FACE_ORDER for suffix in VALID_SUFFIXES
)

