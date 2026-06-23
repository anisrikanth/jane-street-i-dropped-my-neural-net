# Jane Street: I Dropped My Neural Net

This repo contains my solution to Jane Street’s **“I Dropped My Neural Net”** puzzle.

The puzzle gives you a neural network that has been broken into 97 shuffled linear layer files. The goal is to reconstruct the correct order of the model pieces so that the rebuilt network matches the provided historical predictions. The final output is a permutation of the numbers 1-97.

I wrote a full explanation of my approach here:

**[Full writeup: I dropped a neural net](https://arkved.bearblog.dev/i-dropped-a-neural-net/)**

## Overview

The provided model consists of residual blocks and a final linear layer. The 97 shuffled pieces consists of:

- 48 input projection layers
- 48 output projection layers
- 1 final layer

My solution uses the structure of the model itself to reduce the search space:

1. classify pieces by tensor shape
2. pair residual block halves using matrix structure
3. use the Hungarian algorithm to find the best global pairing
4. order the recovered blocks using a norm-based heuristic
5. refine the order with adjacent-swap local search
6. validate the reconstruction against the provided predictions

The final reconstructed model matched the historical predictions to numerical precision.

## How to run the solver

Install the required dependencies:

```bash
pip install numpy pandas torch scipy
```

Then run:

```bash
python dropped_net.py
```

The script loads the shuffled pieces, reconstructs the model ordering, evaluates the reconstruction against the historical data, and prints the recovered permutation.

## Main ideas

The solver relies on some observations:

- The tensor shapes reveal which layers are input projections, output projections, and the final layer.
- Correctly paired residual block halves have stronger structure in `W_out @ W_in`.
- The historical predictions provide a scoring function for candidate block orderings.
- A good initial ordering can be refined with local swaps until the reconstruction matches the original model.

For the detailed explanation, see the full blog post:

**[I dropped a neural net](https://arkved.bearblog.dev/i-dropped-a-neural-net/)**
