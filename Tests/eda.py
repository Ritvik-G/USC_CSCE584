from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SMART_FILE = DATA_DIR / "device_based_consumption.csv"

# Load the CSV file
df = pd.read_csv(SMART_FILE)

# Extract unique Appliance Types
unique_appliances = df['Appliance Type'].unique()

# Print the unique Appliance Types
print("Unique Appliance Types:")
for appliance in unique_appliances:
    print(appliance)



ESSENTIAL_APPLIANCES = {"Fridge", "Heater", "Air Conditioning", "Lights"}
NECESSARY_APPLIANCES = {"Oven", "Microwave", "TV", "Computer"}
EXPENDABLE_APPLIANCES = {"Washing Machine", "Dishwasher"}
