"""Illustrative figures for the mixture-of-experts contrast note.

Both figures illustrate mechanisms discussed in the note; neither is a measured
experimental result. fig_moe_collapse simulates the two-expert self-reinforcing
gate of Appendix B; fig_moe_charge draws the capacity-race schematic.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def fig_collapse() -> None:
    """Two-expert rich-get-richer dynamics (Appendix B), with and without a balancing force."""
    rng = np.random.default_rng(0)
    steps = 400
    lr = 0.25

    def run(balance: float) -> np.ndarray:
        # Expert abilities improve with use and go stale without it; the gate follows the gap.
        z = 0.0  # gate logit difference (expert 1 minus expert 2)
        a = np.array([0.0, 0.0])  # abilities
        decay = 0.02  # staleness of the unselected expert (Appendix B)
        p_hist = np.zeros(steps)
        for t in range(steps):
            p1 = 1.0 / (1.0 + np.exp(-z))
            pick = rng.random() < p1
            i = 0 if pick else 1
            a[i] += lr * (1.0 - a[i])  # the selected expert improves
            a[1 - i] *= 1.0 - decay  # the unselected expert goes stale
            z += lr * (a[0] - a[1]) - balance * (p1 - 0.5)  # gate follows the gap; optional balancing pull
            p_hist[t] = 1.0 / (1.0 + np.exp(-z))
        return p_hist

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(run(balance=0.0), color="tab:red", lw=2, label="no balancing force")
    ax.plot(run(balance=0.35), color="tab:blue", lw=2, label="with balancing force")
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("training step")
    ax.set_ylabel("selection probability of expert 1")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False, loc="center right")
    ax.set_title("Self-reinforcing gate: a two-expert illustration")
    fig.tight_layout()
    fig.savefig("figures/fig_moe_collapse.png", dpi=200)
    plt.close(fig)


def fig_charge() -> None:
    """The capacity race, schematically: growing in-training gain vs. flat charge vs. held-out fit."""
    c = np.linspace(1, 8, 200)
    in_gain = 0.5 * (c - 1)  # ~half a nat per spurious parameter
    prior = np.full_like(c, 1.2)  # a fixed charge
    heldout = 1.6 * (1 - np.exp(-(c - 1) / 1.2)) - 0.22 * np.maximum(c - 3.2, 0.0)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(c, in_gain, color="tab:red", lw=2, label="training-fit gain of extra capacity")
    ax.plot(c, prior, color="tab:orange", lw=2, ls="--", label="fixed in-training charge")
    ax.plot(c, heldout, color="tab:blue", lw=2, label="held-out fit")
    ax.axvline(c[np.argmax(heldout)], color="tab:blue", lw=0.8, ls=":")
    ax.set_xlabel("active capacity")
    ax.set_ylabel("log-likelihood units (schematic)")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title("Why a fixed charge cannot select capacity (schematic)")
    fig.tight_layout()
    fig.savefig("figures/fig_moe_charge.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_collapse()
    fig_charge()
    print("wrote figures/fig_moe_collapse.png, figures/fig_moe_charge.png")
