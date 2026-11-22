"""Task names shared by simulation entry points."""
from .lift import Lift
from .sort import Sort
from .stack import Stack

TASKS = {"lift": Lift, "stack": Stack, "sort": Sort}
