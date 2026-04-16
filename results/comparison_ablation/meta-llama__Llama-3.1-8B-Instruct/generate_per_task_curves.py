from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "per_task_confidence_intervals.csv"
OUTPUT_PNG = BASE_DIR / "per_task_accuracy_curves.png"
SEPARATE_OUTPUT_DIR = BASE_DIR / "per_task_accuracy_charts"
MAIN_METHOD_NAME = "QRScore-SEC"
K_ORDER = [0, 8, 16, 32, 48, 64, 96, 128]

TASK_ORDER = [
    "registrant_name",
    "headquarters_city",
    "headquarters_state",
    "incorporation_state",
    "incorporation_year",
    "employees_count_total",
    "ceo_lastname",
    "holder_record_amount",
]

TASK_LABELS = {
    "registrant_name": "Registrant Name",
    "headquarters_city": "HQ City",
    "headquarters_state": "HQ State",
    "incorporation_state": "Incorp. State",
    "incorporation_year": "Incorp. Year",
    "employees_count_total": "Employees",
    "ceo_lastname": "CEO Last Name",
    "holder_record_amount": "Holder Record Amount",
}


def add_confidence_band(ax, task_df, color):
    ax.fill_between(
        task_df["K"],
        task_df["ci_lo"],
        task_df["ci_hi"],
        color=color,
        alpha=0.12,
    )


def save_combined_chart(df, present_tasks, palette):
    plt.figure(figsize=(12, 7))
    ax = sns.lineplot(
        data=df,
        x="K",
        y="accuracy",
        hue="task",
        hue_order=present_tasks,
        palette=palette,
        marker="o",
        linewidth=2.5,
        markersize=8,
    )

    for color, task in zip(palette, present_tasks):
        task_df = df[df["task"] == task].sort_values("K")
        add_confidence_band(ax, task_df, color)

    ax.set_title("Per-Task Accuracy vs. Heads Ablated (QRScore-SEC)", pad=14, weight="bold")
    ax.set_xlabel("Number of Top Heads Ablated (K)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(K_ORDER)

    handles, labels = ax.get_legend_handles_labels()
    pretty_labels = [TASK_LABELS.get(label, label) for label in labels]
    ax.legend(
        handles,
        pretty_labels,
        title="SEC Extraction Task",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=True,
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUTPUT_PNG.name} successfully!")


def save_separate_task_charts(df, present_tasks, palette):
    SEPARATE_OUTPUT_DIR.mkdir(exist_ok=True)

    for color, task in zip(palette, present_tasks):
        task_df = df[df["task"] == task].sort_values("K")
        label = TASK_LABELS.get(task, task)

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.lineplot(
            data=task_df,
            x="K",
            y="accuracy",
            marker="o",
            linewidth=2.5,
            markersize=8,
            color=color,
            ax=ax,
        )
        add_confidence_band(ax, task_df, color)

        ax.set_title(f"{label}: Accuracy vs. Heads Ablated", pad=12, weight="bold")
        ax.set_xlabel("Number of Top Heads Ablated (K)")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(K_ORDER)
        ax.grid(True, alpha=0.3)

        out_path = SEPARATE_OUTPUT_DIR / f"{task}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path.name} successfully!")


def main():
    df = pd.read_csv(INPUT_CSV)

    df = df[df["method"] == MAIN_METHOD_NAME].copy()
    if df.empty:
        raise ValueError(f"No rows found for method={MAIN_METHOD_NAME!r} in {INPUT_CSV}")

    df["K"] = pd.to_numeric(df["K"])
    df = df[df["K"].isin(K_ORDER)].copy()

    present_tasks = [task for task in TASK_ORDER if task in df["task"].unique()]
    df["task"] = pd.Categorical(df["task"], categories=present_tasks, ordered=True)
    df = df.sort_values(["task", "K"])

    sns.set_theme(style="whitegrid", context="talk")
    palette = sns.color_palette("tab10", n_colors=len(present_tasks))
    save_combined_chart(df, present_tasks, palette)
    save_separate_task_charts(df, present_tasks, palette)


if __name__ == "__main__":
    main()