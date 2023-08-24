import numpy as np
from collections import namedtuple
from math import ceil, floor
from optional import Optional
    

# def forward(x):
#     print("{output}=forward({input})".format(output = x+1, input=x, index=i))
#     return x + 1

# def backward(x, dy):
#     print("{output}=backward({input1}, {input2})".format(output = dy - 1, input1=x, input2=dy, index=i))
#     return dy - 1
    

def nextCheckpoint(total_steps, begin_step, end_step, num_checkpoints):
    step_diff = floor(total_steps / num_checkpoints)
    if begin_step == 0:
        i = 1
    else:
        i = ceil(begin_step / step_diff)
    if i <= (num_checkpoints - 1):
        return Optional.of(i * step_diff)
    else:
        return Optional.empty()

# Uniform spaced checkpointing
def reverseLoopCheckpointed(forward, x0, backward, dy0, num_checkpoints, loop_bound):
    dy = dy0
    
    cache = np.zeros((num_checkpoints, len(x0)))
    Checkpoint = namedtuple("Checkpoint", ["cache_index", "step_index"])
    activeCheckpoints = []    

    activeCheckpoints.append(Checkpoint(0,0))
    cache[activeCheckpoints[-1].cache_index] = x
    print("checkpoint({step_index}/{cache_index})={value}".format(step_index=activeCheckpoints[-1].step_index, cache_index=activeCheckpoints[-1].cache_index, value=cache[activeCheckpoints[-1].cache_index]))

    for i in reversed(range(loop_bound)):
        x = cache[activeCheckpoints[-1].cache_index, :]
        startStep = activeCheckpoints[-1].step_index + 1
        endStep = i + 1

        nextCheckpointStep = nextCheckpoint(loop_bound, startStep, endStep, num_checkpoints)
                
        for j in range(startStep, endStep):
            
            x = forward(x)

            if nextCheckpointStep and (j == nextCheckpointStep.get()):
                
                activeCheckpoints.append(Checkpoint(activeCheckpoints[-1].cache_index + 1, j))
                cache[activeCheckpoints[-1].cache_index, :] = x
                print("checkpoint({step_index}/{cache_index})={value}".format(step_index=activeCheckpoints[-1].step_index, cache_index=activeCheckpoints[-1].cache_index, value=cache[activeCheckpoints[-1].cache_index]))
                
                nextCheckpointStep = nextCheckpoint(loop_bound, j + 1, endStep, num_checkpoints)

        dy = backward(x, dy)
        if i == activeCheckpoints[-1].step_index:
            print("release checkpoint({step_index}/{cache_index})={value}".format(step_index=activeCheckpoints[-1].step_index, cache_index=activeCheckpoints[-1].cache_index, value=cache[activeCheckpoints[-1].cache_index]))
            activeCheckpoints.pop()
    
    return dy

# No checkpointing
def reverseLoop(forward, x0, backward, dy0, num_checkpoints, loop_bound):
    dy = dy0

    for i in reversed(range(loop_bound)):
        x = x0
        startStep = 1
        endStep = i + 1
                
        for j in range(startStep, endStep):
            
            x = forward(x)

        dy = backward(x, dy)
    
    return dy

    