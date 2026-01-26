import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import load

# from torch.nn.functional import softmax
from tqdm import tqdm


def visualize_logits(logits, count):
    plt.plot(logits)
    plt.savefig(f"logits_{count}.png")
    plt.close()


def calculate_entropy_from_logits(
    logits_list: list[torch.Tensor],
) -> list[float]:
    """
    Calculate entropy for each logit vector in a generation sequence.

    Args:
        logits_list: List of logit tensors, one per generated token.
                     Each tensor has shape (vocab_size,) or (1, vocab_size).

    Returns:
        List of entropy values (in nats). Lower = more confident.
    """
    entropies = []
    for logits in logits_list:
        # Ensure 1D
        logits = logits.squeeze()

        # Convert logits to probabilities via softmax
        probs = F.softmax(logits, dim=-1)

        # Calculate entropy: H = -Σ p(x) * log(p(x))
        # Use log_softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs).item()

        entropies.append(entropy)

    avg_entropy = torch.mean(torch.tensor(entropies))
    return avg_entropy


def main():
    logits_list = load("logits.pt")
    count = 0

    for count, logits_list in enumerate(tqdm(logits_list)):
        avg_entropy = calculate_entropy_from_logits(logits_list)
        print(f"Round {count}: Average entropy: {avg_entropy.item()}")


if __name__ == "__main__":
    main()
