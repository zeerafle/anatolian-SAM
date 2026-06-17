import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import wasserstein_distance


def hz_to_cents(f_hz, f_ref=440.0):
    """Convert frequency in Hz to cents relative to a reference frequency."""
    return 1200.0 * np.log2(np.clip(f_hz, a_min=1e-6, a_max=None) / f_ref)


def extract_f0_pyin(
    y, sr, fmin=librosa.note_to_hz("C3"), fmax=librosa.note_to_hz("C6")
):
    """Extract F0 using librosa.pyin, filtering out unvoiced frames."""
    f0, voiced_flag, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr)
    f0_voiced = f0[voiced_flag & ~np.isnan(f0)]
    return f0_voiced


def plot_and_evaluate_3way(
    cents_gt, cents_zs, cents_lora, title, save_path, save=False
):
    """Plot histograms and compute Earth Mover's Distance against ground truth."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True, sharex=True)
    all_cents = (
        np.concatenate([cents_gt, cents_zs, cents_lora]) if len(cents_gt) else []
    )
    bins = (
        np.arange(int(np.min(all_cents)) - 50, int(np.max(all_cents)) + 50, 5)
        if len(all_cents) > 0
        else 50
    )

    ds_zs = (
        wasserstein_distance(cents_gt, cents_zs)
        if len(cents_gt) and len(cents_zs)
        else float("inf")
    )
    ds_lora = (
        wasserstein_distance(cents_gt, cents_lora)
        if len(cents_gt) and len(cents_lora)
        else float("inf")
    )

    sns.histplot(cents_gt, bins=bins, kde=True, ax=axes[0], color="blue", alpha=0.7)
    axes[0].set_title("Ground Truth (Target)")
    axes[0].set_xlabel("Pitch (Cents r.t. 440Hz)")

    sns.histplot(cents_zs, bins=bins, kde=True, ax=axes[1], color="red", alpha=0.7)
    axes[1].set_title(f"Zero-Shot Target\nEMD vs GT: {ds_zs:.1f}")
    axes[1].set_xlabel("Pitch (Cents r.t. 440Hz)")

    sns.histplot(cents_lora, bins=bins, kde=True, ax=axes[2], color="green", alpha=0.7)
    axes[2].set_title(f"LoRA Target\nEMD vs GT: {ds_lora:.1f}")
    axes[2].set_xlabel("Pitch (Cents r.t. 440Hz)")

    # 12-TET markers
    for ax in axes:
        if len(all_cents) > 0:
            min_c, max_c = int(np.min(all_cents)), int(np.max(all_cents))
            for tick in range((min_c // 100) * 100, max_c + 100, 100):
                ax.axvline(tick, color="grey", linestyle="--", alpha=0.3)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save:
        plt.savefig(f"{save_path}/{title}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def compute_emd_metrics(audio_gt, audio_zs, audio_lora, sr):
    """Computes the Earth Mover's Distance without plotting."""
    # 1. Extract F0
    f0_gt = extract_f0_pyin(audio_gt, sr)
    f0_zs = extract_f0_pyin(audio_zs, sr)
    f0_lora = extract_f0_pyin(audio_lora, sr)

    # 2. Convert to Cents
    cents_gt = hz_to_cents(f0_gt)
    cents_zs = hz_to_cents(f0_zs)
    cents_lora = hz_to_cents(f0_lora)

    # 3. Calculate EMD
    emd_zs = (
        wasserstein_distance(cents_gt, cents_zs)
        if len(cents_gt) and len(cents_zs)
        else np.nan
    )
    emd_lora = (
        wasserstein_distance(cents_gt, cents_lora)
        if len(cents_gt) and len(cents_lora)
        else np.nan
    )

    return emd_zs, emd_lora, (cents_gt, cents_zs, cents_lora)
