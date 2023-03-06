#!/usr/bin/env python3
"""
Early stopping related stuffs.

author:
"""



class Monitor:
    """
    The monitor object for early stopping.
    """
    def __init__(self,
        type:str = 'CONTI',
        tolerance:int = 5,
        min_delta:float = 0.0
        ):
        self.type = type.upper()
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0

    def __call__(self, train_loss, validation_loss):
        # Probably we should add counter when change lower than minimum delta
        # if (validation_loss - train_loss) > self.min_delta:
        if abs(validation_loss - train_loss) <= self.min_delta:
            self.counter +=1
        else:
            # Reset counter if not meeting criteria continuously.
            if type == 'CONTI':
                self.counter = 0
            elif type == 'ACCUM':
                continue

        return self.counter >= self.tolerance
