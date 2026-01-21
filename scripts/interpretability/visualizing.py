import matplotlib.pyplot as plt
from torch import load

# from torch.nn.functional import softmax
from tqdm import tqdm


def main():
    logits_list = load("logits.pt")
    count = 0

    for logits in tqdm(logits_list):
        # probabilities = softmax(l, dim=-1).numpy()
        np_logits = logits.numpy()
        # plot the logits
        plt.plot(np_logits)
        plt.savefig(f"logits_{count}.png")
        plt.close()
        if count == 4:
            break
        count += 1


if __name__ == "__main__":
    main()
