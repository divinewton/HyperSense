from audit_binned_common import run_binned_audit


if __name__ == "__main__":
    run_binned_audit(
        metric_label="Basal Metabolic Rate (BMR)",
        type_token="BasalEnergyBurned",
        valid_min=0,
        mode="interval",
    )
