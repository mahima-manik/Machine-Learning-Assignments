from read_data import *

class Tree_Node:
    def __init__(lc, rc):
        self.lc = lc
        self.rc = rc


class Decision_Tree:
    def __init__(root):
        self.root = root



print("The sizes are " , "Train:" , train_data.shape , ", Validation:" , (valid_data.shape) , ", Test:" , len(test_data))