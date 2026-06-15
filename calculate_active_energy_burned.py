from audit_binned_common import run_binned_audit


if __name__ == "__main__":
    run_binned_audit(
        metric_label="Active Energy Burned (Calories)",
        type_token="ActiveEnergyBurned",
        valid_min=0,
        mode="interval",
    )
