import os
import glob
import hashlib

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment


PIECES_DIR = "pieces"
DATA_PATH = "historical_data.csv"

EXPECTED_HASH = "093be1cf2d24094db903cbc3e8d33d306ebca49c6accaa264e44b0b675e7d9c4"


def load_piece(path):
    """
    Load one PyTorch Linear layer state_dict.
    Expected keys: "weight", "bias".
    """
    sd = torch.load(path, map_location="cpu")

    W = sd["weight"].detach().cpu().float().numpy()
    b = sd["bias"].detach().cpu().float().numpy()

    return W, b


def load_pieces():
    """
    Separate the 97 pieces into:
      - 48 input projections:  Linear(48 -> 96), weight shape (96, 48)
      - 48 output projections: Linear(96 -> 48), weight shape (48, 96)
      - 1 final layer:         Linear(48 -> 1),  weight shape (1, 48)
    """
    inp = []
    out = []
    final = None

    paths = sorted(
        glob.glob(os.path.join(PIECES_DIR, "piece_*.pth")),
        key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]),
    )

    print("Looking in:", os.path.abspath(PIECES_DIR))
    print("Found piece files:", len(paths))

    if len(paths) == 0:
        raise FileNotFoundError(
            f"No piece_*.pth files found in {os.path.abspath(PIECES_DIR)}"
        )

    for path in paths:
        idx = int(os.path.basename(path).split("_")[1].split(".")[0])
        W, b = load_piece(path)

        if W.shape == (96, 48):
            inp.append((idx, W, b))
        elif W.shape == (48, 96):
            out.append((idx, W, b))
        elif W.shape == (1, 48):
            final = (idx, W, b)
        else:
            raise ValueError(f"Unexpected shape for piece_{idx}: {W.shape}")

    if len(inp) != 48:
        raise ValueError(f"Expected 48 inp layers, found {len(inp)}")

    if len(out) != 48:
        raise ValueError(f"Expected 48 out layers, found {len(out)}")

    if final is None:
        raise ValueError("Could not find final layer with shape (1, 48)")

    print("Final layer:", final[0])
    print("Input pieces:", len(inp))
    print("Output pieces:", len(out))

    return inp, out, final


def load_data():
    df = pd.read_csv(DATA_PATH)

    feature_cols = [f"measurement_{i}" for i in range(48)]

    missing = [c for c in feature_cols + ["pred"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    X = df[feature_cols].values.astype(np.float32)
    y = df["pred"].values.astype(np.float32)

    return X, y


def pair_score(W_in, W_out):
    """
    Score whether W_in and W_out belong to the same residual block.

    A block has approximately:
        x -> x + W_out ReLU(W_in x + b_in) + b_out

    For the correct pair, W_out @ W_in tends to have much stronger
    diagonal / structured signal than incorrect pairings.
    """
    M = W_out @ W_in  # shape: (48, 48)

    diag = np.diag(M)
    off_diag = M - np.diag(diag)

    diag_strength = np.sum(np.abs(diag))
    off_diag_strength = np.linalg.norm(off_diag)

    return diag_strength / (off_diag_strength + 1e-12)


def recover_pairs(inp, out):
    """
    Recover which input projection goes with which output projection.

    Uses Hungarian assignment to maximize total pair score.
    """
    n = len(inp)
    score = np.zeros((n, n), dtype=np.float64)

    for i, (_, W_in, _) in enumerate(inp):
        for j, (_, W_out, _) in enumerate(out):
            score[i, j] = pair_score(W_in, W_out)

    rows, cols = linear_sum_assignment(-score)

    pair = {}
    for i, j in zip(rows, cols):
        pair[i] = j

    print("\nRecovered pairs:")
    for i in range(n):
        print(f"{inp[i][0]} -> {out[pair[i]][0]}  score={score[i, pair[i]]:.6f}")

    return pair


def apply_block(X, block_idx, inp, out, pair):
    """
    Apply one recovered residual block.
    """
    _, W_in, b_in = inp[block_idx]
    _, W_out, b_out = out[pair[block_idx]]

    H = X @ W_in.T + b_in
    H = np.maximum(H, 0.0)

    return X + H @ W_out.T + b_out


def predict_from_order(X, order, inp, out, pair, final):
    """
    Run the full reconstructed network for a candidate block order.
    """
    Z = X.copy()

    for block_idx in order:
        Z = apply_block(Z, block_idx, inp, out, pair)

    _, W_final, b_final = final
    pred = Z @ W_final.T + b_final

    return pred.reshape(-1)


def mse(pred, y):
    return float(np.mean((pred - y) ** 2))


def order_loss(order, inp, out, pair, final, X, y):
    pred = predict_from_order(X, order, inp, out, pair, final)
    return mse(pred, y)


def initial_order_by_out_norm(pair, inp, out):
    """
    Seed the block order by Frobenius norm of the paired output projection.
    """
    scored = []

    for block_idx in range(len(inp)):
        out_idx = pair[block_idx]
        W_out = out[out_idx][1]

        score = np.linalg.norm(W_out)
        scored.append((score, block_idx))

    order = [block_idx for score, block_idx in sorted(scored)]

    print("\nInitial order by output-layer Frobenius norm:")
    print([inp[i][0] for i in order])

    return order


def adjacent_swap_search(order, inp, out, pair, final, X, y, rounds=200):
    """
    Repeatedly try adjacent swaps.

    Good when the norm-seeded order is mostly correct but has local inversions.
    """
    order = list(order)
    best = order_loss(order, inp, out, pair, final, X, y)

    print(f"\nStarting adjacent-swap search. Initial MSE: {best:.12g}")

    n = len(order)

    for r in range(rounds):
        improved = False

        for i in range(n - 1):
            cand = order.copy()
            cand[i], cand[i + 1] = cand[i + 1], cand[i]

            loss = order_loss(cand, inp, out, pair, final, X, y)

            if loss < best:
                order = cand
                best = loss
                improved = True

                print(
                    f"adj round={r:03d} swap=({i},{i + 1}) "
                    f"mse={best:.12g}"
                )

        if not improved:
            print(f"Adjacent search converged after round {r}.")
            break

    return order, best


def arbitrary_swap_search(order, inp, out, pair, final, X, y, rounds=50):
    """
    Try all pairwise block swaps.
    """
    order = list(order)
    best = order_loss(order, inp, out, pair, final, X, y)

    print(f"\nStarting arbitrary-swap search. Initial MSE: {best:.12g}")

    n = len(order)

    for r in range(rounds):
        improved = False

        for i in range(n):
            for j in range(i + 1, n):
                cand = order.copy()
                cand[i], cand[j] = cand[j], cand[i]

                loss = order_loss(cand, inp, out, pair, final, X, y)

                if loss < best:
                    order = cand
                    best = loss
                    improved = True

                    print(
                        f"swap round={r:03d} swap=({i},{j}) "
                        f"mse={best:.12g}"
                    )

        if not improved:
            print(f"Arbitrary swap search converged after round {r}.")
            break

    return order, best


def three_cycle_search(order, inp, out, pair, final, X, y, rounds=10):
    """
    Try 3-cycles.
    """
    order = list(order)
    best = order_loss(order, inp, out, pair, final, X, y)

    print(f"\nStarting 3-cycle search. Initial MSE: {best:.12g}")

    n = len(order)

    for r in range(rounds):
        improved = False

        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    base = order

                    # Rotation 1:
                    # i <- k, j <- i, k <- j
                    cand1 = base.copy()
                    cand1[i], cand1[j], cand1[k] = base[k], base[i], base[j]

                    loss1 = order_loss(cand1, inp, out, pair, final, X, y)

                    if loss1 < best:
                        order = cand1
                        best = loss1
                        improved = True

                        print(
                            f"cycle round={r:03d} cycle=({i},{j},{k}) dir=1 "
                            f"mse={best:.12g}"
                        )

                        continue

                    # Rotation 2:
                    # i <- j, j <- k, k <- i
                    cand2 = base.copy()
                    cand2[i], cand2[j], cand2[k] = base[j], base[k], base[i]

                    loss2 = order_loss(cand2, inp, out, pair, final, X, y)

                    if loss2 < best:
                        order = cand2
                        best = loss2
                        improved = True

                        print(
                            f"cycle round={r:03d} cycle=({i},{j},{k}) dir=2 "
                            f"mse={best:.12g}"
                        )

        if not improved:
            print(f"3-cycle search converged after round {r}.")
            break

    return order, best


def reverse_order_if_better(order, inp, out, pair, final, X, y):
    """
    Check whether ascending or descending norm order is better.
    """
    forward_loss = order_loss(order, inp, out, pair, final, X, y)

    rev = list(reversed(order))
    reverse_loss = order_loss(rev, inp, out, pair, final, X, y)

    print(f"\nAscending norm MSE:  {forward_loss:.12g}")
    print(f"Descending norm MSE: {reverse_loss:.12g}")

    if reverse_loss < forward_loss:
        print("Using descending norm order.")
        return rev

    print("Using ascending norm order.")
    return order


def make_permutation(order, inp, out, pair, final):
    """
    Convert block order into the required 97-piece permutation:
        inp_0, out_0, inp_1, out_1, ..., inp_47, out_47, final
    """
    perm = []

    for block_idx in order:
        inp_piece = inp[block_idx][0]
        out_piece = out[pair[block_idx]][0]
        perm.extend([inp_piece, out_piece])

    perm.append(final[0])
    return perm


def sha256_perm(perm):
    canonical = ",".join(map(str, perm))
    return hashlib.sha256(canonical.encode()).hexdigest()


def print_solution(order, loss, inp, out, pair, final):
    perm = make_permutation(order, inp, out, pair, final)
    h = sha256_perm(perm)

    print("\nFinal block order as pairs:")
    for block_idx in order:
        print(f"({inp[block_idx][0]}, {out[pair[block_idx]][0]})")

    print("\nFinal MSE:")
    print(loss)

    print("\nPermutation:")
    print(",".join(map(str, perm)))

    print("\nHash:")
    print(h)

    if h == EXPECTED_HASH:
        print("\nSOLVED")
    else:
        print("\nNot solved yet.")
        print("Expected hash:")
        print(EXPECTED_HASH)

    return perm, h


def main():
    inp, out, final = load_pieces()
    X, y = load_data()

    pair = recover_pairs(inp, out)

    order = initial_order_by_out_norm(pair, inp, out)

    # Try both ascending and descending norm order.
    order = reverse_order_if_better(order, inp, out, pair, final, X, y)

    start_loss = order_loss(order, inp, out, pair, final, X, y)
    print(f"\nInitial seeded MSE: {start_loss:.12g}")

    # Local refinement.
    order, loss = adjacent_swap_search(
        order,
        inp,
        out,
        pair,
        final,
        X,
        y,
        rounds=200,
    )

    if loss > 1e-10:
        order, loss = three_cycle_search(
        order,
        inp,
        out,
        pair,
        final,
        X,
        y,
        rounds=10,
    )
    else:
        print("Already solved; skipping 3-cycle search.")

    print_solution(order, loss, inp, out, pair, final)

    # One more cleanup pass
    order, loss = adjacent_swap_search(
        order,
        inp,
        out,
        pair,
        final,
        X,
        y,
        rounds=100,
    )

    order, loss = arbitrary_swap_search(
        order,
        inp,
        out,
        pair,
        final,
        X,
        y,
        rounds=20,
    )

    print_solution(order, loss, inp, out, pair, final)


if __name__ == "__main__":
    main()