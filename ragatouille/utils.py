import os, sys
import io
import random


def seeded_shuffle(collection: list, seed: int = 42):
    random.seed(seed)
    random.shuffle(collection)
    return collection