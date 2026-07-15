import os
import re
import sys
import pandas as pd

TIME_OF_DAY_ORDER = ["Morning", "Midday", "Afternoon"]


def resolve_mocopi_dir():
    for path in (
        "/Users/divinewton/Downloads/MOCOPI",
        "/Users/divinewton/Documents/MOCOPI",
    ):
        if os.path.exists(path):
            return path
    return None


def format_mean_sd(values) -> str:
    series = pd.Series(values).dropna()
    if series.empty:
        return "N/A"
    if len(series) == 1:
        return f"{series.iloc[0]:.2f}"
    return f"{series.mean():.2f} ± {series.std(ddof=1):.2f}"


def format_range(values) -> str:
    series = pd.Series(values).dropna()
    if series.empty:
        return "N/A"
    return f"{series.min():.2f} -- {series.max():.2f}"


def pooled_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    summary = df.groupby(group_col, observed=False).agg(
        Intensity=("Intensity", "mean"),
        Variability=("Variability", "mean"),
        Jerk=("Jerk", "mean"),
        Active_Pct=("Is_Active", lambda x: x.mean() * 100),
    )
    if group_col == "Time_of_Day":
        summary = summary.reindex(TIME_OF_DAY_ORDER)
    return summary.round(2)


def participant_means(df: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    group_keys = ["Participant"] if group_col is None else ["Participant", group_col]
    return (
        df.groupby(group_keys, observed=False)
        .agg(
            Intensity=("Intensity", "mean"),
            Variability=("Variability", "mean"),
            Jerk=("Jerk", "mean"),
            Active_Pct=("Is_Active", lambda x: x.mean() * 100),
        )
        .reset_index()
    )


def participant_variation_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    per_participant = participant_means(df, group_col)
    rows = []
    for group_value, group_df in per_participant.groupby(group_col, observed=False):
        rows.append(
            {
                group_col: group_value,
                "Intensity": format_mean_sd(group_df["Intensity"]),
                "Variability": format_mean_sd(group_df["Variability"]),
                "Jerk": format_mean_sd(group_df["Jerk"]),
                "Active_Pct": format_mean_sd(group_df["Active_Pct"]),
            }
        )
    summary = pd.DataFrame(rows)
    if group_col == "Time_of_Day":
        summary["Time_of_Day"] = pd.Categorical(
            summary["Time_of_Day"], categories=TIME_OF_DAY_ORDER, ordered=True
        )
        summary = summary.sort_values("Time_of_Day")
    return summary


def print_section(title: str, frame: pd.DataFrame) -> None:
    print("\n" + "=" * 50)
    print(f" {title}")
    print("=" * 50)
    print(frame.to_string(index=False) if isinstance(frame, pd.DataFrame) else frame)


def main():
    mocopi_dir = resolve_mocopi_dir()
    if mocopi_dir is None:
        print("[Fatal Error] Could not find 'MOCOPI' folder in Downloads or Documents.")
        sys.exit(1)

    all_epochs = []
    for folder in sorted(os.listdir(mocopi_dir)):
        if not re.match(r"^P\d+$", folder):
            continue
        folder_path = os.path.join(mocopi_dir, folder)
        features_file = os.path.join(folder_path, f"{folder}_epoch_kinematics.csv")
        if os.path.exists(features_file):
            all_epochs.append(pd.read_csv(features_file))

    if not all_epochs:
        print(
            "[Fatal Error] No epoch kinematics files found. "
            "Run process_and_analyze_mocopi.py first."
        )
        sys.exit(1)

    cohort_df = pd.concat(all_epochs, ignore_index=True)
    print(f"Loaded {len(all_epochs)} participant epoch files from: {mocopi_dir}")

    print_section(
        "GLOBAL COHORT SUMMARY: BY SENSOR PLACEMENT",
        pooled_summary(cohort_df, "Sensor").reset_index(),
    )
    print_section(
        "GLOBAL COHORT SUMMARY: BY CLASSROOM CONTEXT",
        pooled_summary(cohort_df, "class").reset_index(),
    )
    print_section(
        "GLOBAL COHORT SUMMARY: BY TIME OF DAY",
        pooled_summary(cohort_df, "Time_of_Day").reset_index(),
    )

    print_section(
        "GLOBAL COHORT SUMMARY: CROSS-BODY CORRELATIONS",
        cohort_df.pivot_table(
            index=["Participant", "Date", "class", "Epoch_1Min"],
            columns="Sensor",
            values="Intensity",
        )
        .dropna()
        .corr(method="pearson")
        .round(3),
    )

    overall = participant_means(cohort_df)
    print_section(
        "PARTICIPANT-LEVEL VARIATION: OVERALL",
        pd.DataFrame(
            [
                {
                    "Metric": "Intensity",
                    "Mean ± SD": format_mean_sd(overall["Intensity"]),
                    "Range (Min -- Max)": format_range(overall["Intensity"]),
                },
                {
                    "Metric": "Variability",
                    "Mean ± SD": format_mean_sd(overall["Variability"]),
                    "Range (Min -- Max)": format_range(overall["Variability"]),
                },
                {
                    "Metric": "Jerk",
                    "Mean ± SD": format_mean_sd(overall["Jerk"]),
                    "Range (Min -- Max)": format_range(overall["Jerk"]),
                },
                {
                    "Metric": "Active_Pct",
                    "Mean ± SD": format_mean_sd(overall["Active_Pct"]),
                    "Range (Min -- Max)": format_range(overall["Active_Pct"]),
                },
            ]
        ),
    )
    print_section(
        "PARTICIPANT-LEVEL VARIATION: BY SENSOR PLACEMENT",
        participant_variation_summary(cohort_df, "Sensor"),
    )
    print_section(
        "PARTICIPANT-LEVEL VARIATION: BY CLASSROOM CONTEXT",
        participant_variation_summary(cohort_df, "class"),
    )
    print_section(
        "PARTICIPANT-LEVEL VARIATION: BY TIME OF DAY",
        participant_variation_summary(cohort_df, "Time_of_Day"),
    )


if __name__ == "__main__":
    main()
