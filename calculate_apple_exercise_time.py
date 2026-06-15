from audit_binned_common import run_binned_audit


if __name__ == "__main__":
    run_binned_audit(
        metric_label="Logged Exercise Time (Minutes)",
        type_token="AppleExerciseTime",
        valid_min=0,
        mode="event",
    )
