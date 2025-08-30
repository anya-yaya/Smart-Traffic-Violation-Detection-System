# src/database/database_manager.py

import sqlite3
import pandas as pd
import os
from datetime import datetime

DB_PATH = "database.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create Vehicles table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_no TEXT UNIQUE NOT NULL,
            owner_name TEXT,
            model TEXT
        )
    ''')

    # Create Violations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_id INTEGER,
            violation_type TEXT NOT NULL,
            fine_amount REAL NOT NULL,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            image_path TEXT,
            FOREIGN KEY (vehicle_id) REFERENCES vehicles(id)
        )
    ''')
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

def insert_vehicle(vehicle_no, owner_name=None, model=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO vehicles (vehicle_no, owner_name, model) VALUES (?, ?, ?)",
                       (vehicle_no.upper(), owner_name, model))
        conn.commit()
        print(f"Vehicle {vehicle_no} inserted successfully.")
        return cursor.lastrowid # Return the ID of the newly inserted vehicle
    except sqlite3.IntegrityError:
        print(f"Vehicle {vehicle_no} already exists.")
        cursor.execute("SELECT id FROM vehicles WHERE vehicle_no = ?", (vehicle_no.upper(),))
        return cursor.fetchone()[0] # Return existing vehicle ID
    finally:
        conn.close()

def insert_violation(vehicle_no, violation_type, fine_amount, image_path=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM vehicles WHERE vehicle_no = ?", (vehicle_no.upper(),))
    vehicle_id_row = cursor.fetchone()

    if vehicle_id_row:
        vehicle_id = vehicle_id_row[0]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO violations (vehicle_id, violation_type, fine_amount, timestamp, image_path) VALUES (?, ?, ?, ?, ?)",
                       (vehicle_id, violation_type, fine_amount, timestamp, image_path))
        conn.commit()
        print(f"Violation '{violation_type}' recorded for {vehicle_no}.")
    else:
        print(f"Error: Vehicle {vehicle_no} not found. Cannot record violation.")
    conn.close()

def get_vehicle_info(vehicle_no):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT vehicle_no, owner_name, model FROM vehicles WHERE vehicle_no = ?", (vehicle_no.upper(),))
    vehicle_data = cursor.fetchone()

    violations_data = []
    if vehicle_data:
        cursor.execute('''
            SELECT v.violation_type, v.fine_amount, v.timestamp
            FROM violations v
            JOIN vehicles veh ON v.vehicle_id = veh.id
            WHERE veh.vehicle_no = ?
            ORDER BY v.timestamp DESC
        ''', (vehicle_no.upper(),))
        violations_data = cursor.fetchall()
    conn.close()
    return vehicle_data, violations_data

def get_all_violations():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM violations", conn)
    conn.close()
    return df

def execute_query(query):
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        raise e
    finally:
        conn.close()

class DatabaseManager: # Wrapper class if needed, though direct functions are used here
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        init_db() # Ensure DB is initialized on object creation (or call it once globally)

    def insert_vehicle(self, vehicle_no, owner_name=None, model=None):
        return insert_vehicle(vehicle_no, owner_name, model)

    def insert_violation(self, vehicle_no, violation_type, fine_amount, image_path=None):
        insert_violation(vehicle_no, violation_type, fine_amount, image_path)

    def get_vehicle_info(self, vehicle_no):
        return get_vehicle_info(vehicle_no)

    def get_all_violations(self):
        return get_all_violations()

    def execute_query(self, query):
        return execute_query(query)
