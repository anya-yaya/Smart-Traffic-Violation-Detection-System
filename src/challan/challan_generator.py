# src/challan/challan_generator.py

from src.database.database_manager import insert_violation # Import database manager
import datetime

# Define fine amounts for different violations
FINE_AMOUNTS = {
    "Not Wearing Helmet": 500,
    "Triple Riding": 1000,
    "Using Phone While Riding": 1500,
    # Add more violations and their fines
}

class ChallanGenerator:
    def __init__(self):
        pass

    def generate(self, vehicle_no, violations_list, image_path=None):
        total_fine = 0
        for violation in violations_list:
            fine = FINE_AMOUNTS.get(violation, 0) # Get fine, default to 0 if not found
            total_fine += fine
            # Store each violation separately in the database
            insert_violation(vehicle_no, violation, fine, image_path)
        return total_fine