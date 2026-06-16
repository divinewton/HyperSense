from Coverage.audit_binned_common import run_binned_audit


if __name__ == "__main__":
    run_binned_audit(
        metric_label="Heart Rate",
        type_token="HeartRate",
        valid_min=40,
        valid_max=180,
        mode="point",
    )
