#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

import datasets
import models.models_factory as models_factory
import config

from utils import get_classwise_ds, build_retain_sets_in_unlearning


def clean_state_dict(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def extract_method_name(weight_path):

    path = weight_path.lower()

    if "amnesiac" in path:
        return "Amnesiac"
    elif "finetune" in path:
        return "Finetune"
    elif "retrain" in path:
        return "Retrain"
    elif "scrub" in path:
        return "Scrub"
    elif "bad_t" in path:
        return "Bad-T"
    elif "fdcr" in path:
        return "FDCR"
    elif "l_codec" in path:
        return "L-CODEC"
    elif "orthogonality" in path:
        return "OUR"
    elif "salun" in path:
        return "Salun"
    elif "unrolling" in path:
        return "Unrolling"
    else:
        return "Unknown"


def rea_relearning(
    model,
    forget_dl,
    ood_dl,
    retain_dl,
    device,
    lr,
    weight_path,
    strategy
):

    threshold = 0.75

    model.load_state_dict(
        clean_state_dict(
            torch.load(weight_path, weights_only=False)
        )
    )

    model.eval()

    # 🔥 Ghost detection
    correct, total = 0, 0

    with torch.no_grad():

        for x, _, y in ood_dl:

            out = model(x.to(device))
            _, pred = out.max(1)

            total += y.size(0)
            correct += pred.eq(y.to(device)).sum().item()

    print(f"[Ghost Detection] Initial OOD Accuracy: {correct / total:.4f}")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9
    )

    def relearn(dataloader):

        model.load_state_dict(
            clean_state_dict(
                torch.load(weight_path, weights_only=False)
            )
        )

        model.train()

        # 🔴 Reinitialize optimizer each time (avoid contamination)
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9
        )

        traj, losses = [], []

        retain_iter = iter(retain_dl)

        for batch_idx, (inputs, _, targets) in enumerate(dataloader):

            try:
                r_inputs, _, r_targets = next(retain_iter)

            except StopIteration:

                retain_iter = iter(retain_dl)
                r_inputs, _, r_targets = next(retain_iter)

            inputs = torch.cat((inputs, r_inputs), dim=0).to(device)
            targets = torch.cat((targets, r_targets), dim=0).to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            # Evaluation
            model.eval()

            correct, total = 0, 0

            for x, _, y in dataloader:

                x, y = x.to(device), y.to(device)

                out = model(x)
                _, pred = out.max(1)

                total += y.size(0)
                correct += pred.eq(y).sum().item()

            traj.append(correct / total)
            losses.append(loss.item())

            model.train()

            if batch_idx > 75:
                break

        return traj, losses

    forget_traj, forget_loss = relearn(forget_dl)
    ood_traj, ood_loss = relearn(ood_dl)

    def find_index(traj):

        for i, acc in enumerate(traj):

            if acc >= threshold:
                return i

        return len(traj) - 1

    return {
        "forget_traj": forget_traj,
        "ood_traj": ood_traj,
        "forget_loss": forget_loss,
        "ood_loss": ood_loss,
        "forget_idx": find_index(forget_traj),
        "ood_idx": find_index(ood_traj)
    }


# =========================
# Metrics
# =========================
def compute_metrics(res):

    f_traj = np.array(res["forget_traj"])
    o_traj = np.array(res["ood_traj"])

    f_loss = np.array(res["forget_loss"])
    o_loss = np.array(res["ood_loss"])

    # 🔴 ΔIL: average of first K steps
    K = 5

    f_init = np.mean(f_loss[:K])
    o_init = np.mean(o_loss[:K])

    deltaIL = float(abs(f_init - o_init))

    # CG
    CG = abs(res["ood_idx"] - res["forget_idx"])

    # SBC
    min_len = min(len(f_traj), len(o_traj))

    SBC = 1 - np.mean(
        np.abs(f_traj[:min_len] - o_traj[:min_len])
    )

    return {
        "DeltaIL": deltaIL,
        "CG": float(CG),
        "SBC": float(SBC)
    }


def strategy_score(m, strategy):

    R_il = min(m["DeltaIL"], 1.0)
    R_cg = min(m["CG"] / 50.0, 1.0)
    R_sbc = 1 - m["SBC"]

    if strategy == "Basic":
        w = (0.5, 0.3, 0.2)

    elif strategy == "ReA":
        w = (0.2, 0.5, 0.3)

    else:
        w = (0.2, 0.3, 0.5)

    return (
        w[0] * R_il +
        w[1] * R_cg +
        w[2] * R_sbc
    )


# =========================
# Visualization
# =========================
def plot_strategy_comparison(results, UFRS, save_path, method_name):

    strategies = ["Basic", "ReA", "Advanced"]

    # --- Global font settings ---
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['axes.unicode_minus'] = False

    # =========================
    # Color configuration
    # =========================
    bg_fig = "#F6F4EF"
    bg_ax = "#FBFAF7"

    spine_color = "#5E748C"
    grid_color = "#CDD6DF"

    text_color = "#1F1F1F"
    threshold_color = "#8F8F8F"

    forget_colors = {
        "Basic": "#D88A2D",
        "ReA": "#8B62C2",
        "Advanced": "#7BAA8B"
    }

    ood_color = "#9B9B9B"
    ood_bar_color = "#CBCBCB"

    fig, axes = plt.subplots(
        2, 3,
        figsize=(18, 11),
        dpi=300
    )

    fig.patch.set_facecolor(bg_fig)

    # =========================
    # Top row: Accuracy curves
    # =========================
    for i, s in enumerate(strategies):

        ax = axes[0, i]

        ax.set_facecolor(bg_ax)

        f_traj = results[s]["forget_traj"]
        o_traj = results[s]["ood_traj"]

        ax.plot(
            o_traj,
            linestyle="-",
            linewidth=3.5,
            color=ood_color,
            label="OOD"
        )

        ax.plot(
            f_traj,
            linestyle="--",
            linewidth=3.5,
            color=forget_colors[s],
            label="Forget"
        )

        ax.axhline(
            0.75,
            color=threshold_color,
            linestyle=":",
            linewidth=1.5,
            alpha=0.8
        )

        ax.set_title(
            f"{s} Strategy",
            fontsize=20,
            fontweight='bold',
            pad=20,
            color=text_color
        )

        ax.set_ylim(0, 1.05)

        ax.set_xlabel(
            "Steps",
            fontsize=18,
            fontweight='bold',
            color=text_color
        )

        if i == 0:
            ax.set_ylabel(
                "Accuracy",
                fontsize=18,
                fontweight='bold',
                color=text_color
            )

        ax.tick_params(
            axis='both',
            which='major',
            labelsize=14,
            width=2,
            length=6,
            colors=text_color
        )

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.spines['left'].set_color(spine_color)
        ax.spines['bottom'].set_color(spine_color)

        ax.spines['left'].set_linewidth(1.8)
        ax.spines['bottom'].set_linewidth(1.8)

        ax.grid(
            True,
            linestyle="--",
            alpha=0.6,
            color=grid_color
        )

        leg = ax.legend(
            fontsize=16,
            frameon=False,
            loc='lower right',
            prop={'weight': 'bold', 'size': 16}
        )

        for text in leg.get_texts():
            text.set_color(text_color)

    # =========================
    # Bottom row: Initial loss bar charts
    # =========================
    for i, s in enumerate(strategies):

        ax = axes[1, i]

        ax.set_facecolor(bg_ax)

        K = 5

        ood_loss = np.mean(results[s]["ood_loss"][:K])
        forget_loss = np.mean(results[s]["forget_loss"][:K])

        labels = ["OOD", "Forget"]
        values = [ood_loss, forget_loss]

        bars = ax.bar(
            labels,
            values,
            color=[ood_bar_color, forget_colors[s]],
            alpha=0.95,
            edgecolor=spine_color,
            linewidth=1.5
        )

        ax.set_title(
            f"{s} Initial Loss",
            fontsize=20,
            fontweight='bold',
            pad=20,
            color=text_color
        )

        if i == 0:
            ax.set_ylabel(
                "Loss",
                fontsize=18,
                fontweight='bold',
                color=text_color
            )

        ax.tick_params(
            axis='y',
            which='major',
            labelsize=14,
            width=2,
            length=6,
            colors=text_color
        )

        ax.set_xticks(range(len(labels)))

        ax.set_xticklabels(
            labels,
            fontsize=18,
            fontweight='heavy',
            color=text_color
        )

        for bar in bars:

            height = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + (max(values) * 0.01),
                f"{height:.2f}",
                ha='center',
                va='bottom',
                fontsize=15,
                fontweight='bold',
                color=text_color
            )

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.spines['left'].set_color(spine_color)
        ax.spines['bottom'].set_color(spine_color)

        ax.spines['left'].set_linewidth(1.8)
        ax.spines['bottom'].set_linewidth(1.8)

        ax.grid(
            axis='y',
            linestyle="--",
            alpha=0.6,
            color=grid_color
        )

    # =========================
    # Layout
    # =========================
    plt.subplots_adjust(
        hspace=0.45,
        wspace=0.28,
        top=0.92,
        bottom=0.08,
        left=0.08,
        right=0.95
    )

    plt.savefig(
        save_path,
        dpi=300,
        facecolor=fig.get_facecolor()
    )

    plt.close()

    print(f"Professional evaluation figure saved: {save_path}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-net", type=str)
    parser.add_argument("-dataset", type=str)

    parser.add_argument("-classes", type=int)
    parser.add_argument("-num_classes", type=int)

    parser.add_argument("-b", type=int, default=64)

    parser.add_argument(
        "--forget_class",
        type=int,
        required=True
    )

    parser.add_argument(
        "--unlearn_data_percent",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "-weight_path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.01
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    method_name = extract_method_name(args.weight_path)

    # Paths
    model_dir = os.path.dirname(args.weight_path)

    save_dir = os.path.join(
        model_dir,
        "rea_enhanced"
    )

    os.makedirs(save_dir, exist_ok=True)

    print("Method:", method_name)
    print("Save path:", save_dir)

    # Dataset
    trainset = getattr(datasets, args.dataset)(
        root="./data",
        train=True,
        download=True,
        unlearning=True,
        img_size=32
    )

    classwise_train = get_classwise_ds(
        trainset,
        args.num_classes
    )

    forget_train = classwise_train[args.forget_class]

    len_data = int(
        len(forget_train) * args.unlearn_data_percent
    )

    forget_train = Subset(
        forget_train,
        np.random.choice(
            len(forget_train),
            len_data,
            replace=False
        )
    )

    if args.dataset == 'Cifar10':
        ood_classes = config.cifar10_ood_classes
    else:
        ood_classes = config.ood_classes

    selected_ood = random.choice(ood_classes)

    ood_train = classwise_train[selected_ood]

    ood_train = Subset(
        ood_train,
        np.random.choice(
            len(ood_train),
            len_data,
            replace=False
        )
    )

    retain_train, _ = build_retain_sets_in_unlearning(
        classwise_train,
        classwise_train,
        args.num_classes,
        args.forget_class,
        ood_classes
    )

    retain_dl = DataLoader(
        retain_train,
        batch_size=16,
        shuffle=True
    )

    print(f"Forget={len(forget_train)} | OOD={len(ood_train)}")

    strategies = ["Basic", "ReA", "Advanced"]

    results = {}

    for s in strategies:

        print(f"\n===== {s} =====")

        model = getattr(
            models_factory,
            args.net
        )(num_classes=args.classes).to(device)

        def relabel(ds, new_label):
            return [(x, y, new_label) for x, y, c in ds]

        if s == "Basic":

            forget_ds = relabel(
                forget_train,
                selected_ood
            )

            ood_ds = relabel(
                ood_train,
                selected_ood
            )

        elif s == "ReA":

            forget_ds = forget_train

            ood_ds = relabel(
                ood_train,
                args.forget_class
            )

        else:

            forget_ds = forget_train
            ood_ds = ood_train

        forget_dl = DataLoader(
            forget_ds,
            batch_size=args.b,
            shuffle=True
        )

        ood_dl = DataLoader(
            ood_ds,
            batch_size=args.b,
            shuffle=True
        )

        res = rea_relearning(
            model,
            forget_dl,
            ood_dl,
            retain_dl,
            device,
            args.lr,
            args.weight_path,
            s
        )

        results[s] = res

    # Save results
    with open(
        os.path.join(save_dir, "results.json"),
        "w"
    ) as f:
        json.dump(results, f, indent=2)

    # Metrics
    metrics = {}

    for s in strategies:

        m = compute_metrics(results[s])

        m["score"] = strategy_score(m, s)

        metrics[s] = m

    UFRS = 1 - (
        0.2 * metrics["Basic"]["score"] +
        0.3 * metrics["ReA"]["score"] +
        0.5 * metrics["Advanced"]["score"]
    )

    metrics["UFRS"] = float(UFRS)

    with open(
        os.path.join(save_dir, "metrics.json"),
        "w"
    ) as f:
        json.dump(metrics, f, indent=2)

    print("\nUFRS:", UFRS)

    # Plot
    plot_strategy_comparison(
        results,
        UFRS,
        os.path.join(
            save_dir,
            "strategy_comparison_all.png"
        ),
        method_name
    )

    print("Finished ✔")


if __name__ == "__main__":
    main()