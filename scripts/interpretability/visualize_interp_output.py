import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import log_softmax, softmax
from tqdm import tqdm


def visualize_entropies(entropies, filename):
    plt.plot(entropies)
    plt.savefig(filename)
    plt.close()


def visualize_logits(f_logits, s_logits, filename):
    f_probs = softmax(f_logits[5])
    s_probs = softmax(s_logits[5])
    # make transparent
    plt.plot(f_probs, alpha=0.5)
    plt.plot(s_probs, alpha=0.5)
    plt.legend(["f_probs", "s_probs"])
    plt.savefig(filename)
    plt.close()


def calculate_entropy_from_logits(
    logits_list: list[np.ndarray],
) -> float:
    """
    Calculate entropy for each logit vector in a generation sequence.

    Args:
        logits_list: List of logit arrays, one per generated token.
                     Each array has shape (vocab_size,) or (1, vocab_size).

    Returns:
        Average entropy value (in nats). Lower = more confident.
    """
    entropies = []
    for logits in logits_list:
        # Ensure 1D
        logits = np.squeeze(logits)

        # Convert logits to probabilities via softmax
        probs = softmax(logits, axis=-1)

        # Calculate entropy: H = -Σ p(x) * log(p(x))
        # Use log_softmax for numerical stability
        log_probs = log_softmax(logits, axis=-1)
        entropy = -np.sum(probs * log_probs)

        entropies.append(entropy)

    avg_entropy = np.mean(entropies)
    return avg_entropy


def main():
    f_generation_id = 5
    s_generation_id = 25
    df = pd.read_parquet(
        "results/drift_experiment_20260127_085456_interp.parquet"
    )
    entropy_file_name = "avg_entropies3"
    logits_file_name = "logits3"

    # Access the '0' column directly
    if "{entropy_file_name}.parquet" in os.listdir():
        avg_entropies = pd.read_parquet(f"{entropy_file_name}.parquet")
    else:
        avg_entropies = []
        for count, logits_dict in enumerate(tqdm(df["logits"])):
            if logits_dict is not None:
                logits = logits_dict["logits"]
                avg_entropy = calculate_entropy_from_logits(logits)
                avg_entropies.append(avg_entropy)

                print(f"Round {count}: Average entropy: {avg_entropy}")
            else:
                avg_entropies.append(np.nan)
                print(f"No generation for round {count}")
    f_logits = None
    s_logits = None
    for count, logits_dict in enumerate(tqdm(df["logits"])):
        if logits_dict is not None:
            logits = logits_dict["logits"]
            if count == f_generation_id:
                f_logits = logits
            if count == s_generation_id:
                s_logits = logits

    if f_logits is not None and s_logits is not None:
        visualize_logits(f_logits, s_logits, f"{logits_file_name}.png")
    if f"{entropy_file_name}.parquet" not in os.listdir():
        df = pd.DataFrame({"avg_entropy": avg_entropies})
        df.to_parquet(f"{entropy_file_name}.parquet")
    visualize_entropies(avg_entropies, f"{entropy_file_name}.png")


if __name__ == "__main__":
    main()
