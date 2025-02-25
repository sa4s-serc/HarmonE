import random
import datetime

# Cloud regions with cost and sustainability scores
CLOUD_REGIONS = {
    "Amsterdam": {"base_cost": 0.5, "sustainability": 0.9},  # Green but expensive
    "India": {"base_cost": 0.2, "sustainability": 0.5},  # Less green but cheaper
    "USA": {"base_cost": 0.35, "sustainability": 0.7},  # Balanced
}

def get_dynamic_price(region):
    """Simulates hourly price fluctuations in different regions."""
    hour = datetime.datetime.now().hour
    base_cost = CLOUD_REGIONS[region]["base_cost"]

    if 6 <= hour <= 18:  # Daytime: higher demand
        fluctuation = random.uniform(0.2, 0.5)  # 20-50% increase
    else:  # Nighttime: lower demand
        fluctuation = random.uniform(-0.1, 0.2)  # -10% to 20% change

    return round(base_cost * (1 + fluctuation), 3)

def get_sustainability_score(region):
    """Returns the sustainability score of a cloud region."""
    return CLOUD_REGIONS[region]["sustainability"]

if __name__ == "__main__":
    for region in CLOUD_REGIONS:
        print(f"{region}: Price ${get_dynamic_price(region)} per unit, Sustainability Score {get_sustainability_score(region)}")
