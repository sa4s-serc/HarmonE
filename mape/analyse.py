from monitor import monitor_mape, monitor_drift

def analyse():
    mape_data = monitor_mape()
    drift_data = monitor_drift()

    if not mape_data or not drift_data:
        return None

    performance_issue = mape_data["mape"] > 10
    slow_model = mape_data["avg_time"] > 0.01
    high_energy = mape_data["avg_energy"] > 15
    drift_detected = drift_data["kl_div"] > 0.5 or drift_data["energy_distance"] > 0.1
    print(drift_detected)

    return {
        "performance_issue": performance_issue,
        "slow_model": slow_model,
        "high_energy": high_energy,
        "drift_detected": drift_detected
    }
