class Value:
    # stores a single scalar value and its gradient

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self.backward = lambda: None
        
