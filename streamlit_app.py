# streamlit_app.py
import streamlit as st
import time
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tempfile
import os
import numpy as np 
import seaborn as sns  # Added for image conversion

# Import your modules
from src.detection.yolo_detector import ViolationDetector
from src.npr.npr_processor import NumberPlateRecognizer
from src.database.database_manager import init_db, insert_vehicle, get_vehicle_info, execute_query
from src.challan.challan_generator import ChallanGenerator
from src.utils.helpers import save_uploaded_file, clear_temp_directory

# Initialize Database (run once)
init_db()

# Initialize components using Streamlit caching for performance
@st.cache_resource
def load_detector():
    return ViolationDetector()

@st.cache_resource
def load_npr():
    return NumberPlateRecognizer()

@st.cache_resource
def load_challan_gen():
    return ChallanGenerator()

detector = load_detector()
npr = load_npr()
challan_gen = load_challan_gen()

st.title("🚦 Smart Traffic Violation Detection System")
st.sidebar.header("Navigation")
menu = st.sidebar.selectbox("Select Option", ["Detect Violation", "Vehicle Info", "Analytics", "SQL Query"])

if menu == "Detect Violation":
    st.subheader("Upload Media for Violation Detection")
    file = st.file_uploader("Upload image or video", type=["jpg", "png", "jpeg", "mp4", "avi", "mov"])

    if file:
        temp_dir = None # Initialize temp_dir to None
        try:
            file_ext = file.name.split('.')[-1].lower()
            is_video = file_ext in ["mp4", "avi", "mov"]
            original_file_path = save_uploaded_file(file)
            temp_dir = os.path.dirname(original_file_path) # Store the temporary directory

            if not is_video:
                st.image(original_file_path, caption="Uploaded Image", use_column_width=True)

                # Image processing
                violations, annotated_image_cv2 = detector.detect(original_file_path)
                if annotated_image_cv2 is not None:
                    # Convert OpenCV image (BGR) to RGB for Streamlit display
                    annotated_image_rgb = cv2.cvtColor(annotated_image_cv2, cv2.COLOR_BGR2RGB)
                    st.image(annotated_image_rgb, caption="Detected Violations", use_column_width=True)
                else:
                    st.error("Failed to annotate image. Check logs for details.")

                detected_plate = npr.extract_plate(original_file_path)
                st.write(f"**Detected Number Plate:** `{detected_plate}`")

                # Input field for vehicle number (allows correction)
                corrected_plate = st.text_input("Corrected Vehicle Number (if needed):", value=detected_plate)

                st.write(f"**Detected Violations:** {', '.join(violations) if violations else 'None'}")

                # Owner and Model input fields
                owner_name = st.text_input("Owner Name (optional)")
                vehicle_model = st.text_input("Vehicle Model (optional)")

                if st.button("Generate Challan"):
                    if not corrected_plate or corrected_plate == "N/A":
                        st.error("Please enter a valid Vehicle Number to generate a challan.")
                    else:
                        vehicle_id = insert_vehicle(corrected_plate, owner_name, vehicle_model)
                        if violations:
                            # Pass original_file_path to challan generator for image path storage
                            total_fine = challan_gen.generate(corrected_plate, violations, original_file_path)
                            st.success(f"E-Challan Generated for {corrected_plate}. Total Fine: ₹{total_fine}")
                        else:
                            st.info("No violations detected. No challan generated.")
            else:
                # Video processing
                st.info("Processing video... this may take a few moments. Violations will be collected over time.")
                st.warning("Real-time video stream processing not implemented. Analyzing frames.")

                cap = cv2.VideoCapture(original_file_path)
                if not cap.isOpened():
                    st.error("Error: Could not open video file.")
                else:
                    frame_rate = 5  # Process every 5th frame to speed up
                    frame_count = 0
                    all_violations_set = set() # Use a set to store unique violations

                    st_frame = st.empty() # Placeholder for video frames (optional)
                    progress_bar = st.progress(0)

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_count += 1
                        if frame_count % frame_rate == 0:
                            # Save frame temporarily for detection
                            frame_path = os.path.join(temp_dir, f"frame_{frame_count}.jpg")
                            cv2.imwrite(frame_path, frame)

                            current_violations, annotated_frame = detector.detect(frame_path)
                            all_violations_set.update(current_violations)

                            # Optional: Display annotated frame
                            if annotated_frame is not None:
                                st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                                                caption=f"Frame {frame_count}: Detected: {', '.join(current_violations)}",
                                                use_column_width=True)
                                time.sleep(0.05) # Little delay for visual

                            # Only extract plate once or if it's still unknown
                            if 'detected_plate_video' not in st.session_state or st.session_state.detected_plate_video == "N/A":
                                detected_plate_video = npr.extract_plate(frame_path)
                                if detected_plate_video != "N/A":
                                    st.session_state.detected_plate_video = detected_plate_video

                        progress = int(cap.get(cv2.CAP_PROP_POS_FRAMES) / cap.get(cv2.CAP_PROP_FRAME_COUNT) * 100)
                        progress_bar.progress(progress)

                    cap.release()
                    st.success("Video processing complete.")

                    detected_plate_final = st.session_state.get('detected_plate_video', "N/A")
                    st.write(f"**Detected Number Plate:** `{detected_plate_final}`")

                    # Input field for vehicle number (allows correction)
                    corrected_plate_video = st.text_input("Corrected Vehicle Number (if needed):", value=detected_plate_final, key='video_plate_input')

                    st.write(f"**Violations Detected in Video:** {', '.join(all_violations_set) if all_violations_set else 'None'}")

                    owner_name_video = st.text_input("Owner Name (optional)", key='video_owner_input')
                    vehicle_model_video = st.text_input("Vehicle Model (optional)", key='video_model_input')

                    if st.button("Generate Challan for Video"):
                        if not corrected_plate_video or corrected_plate_video == "N/A":
                            st.error("Please enter a valid Vehicle Number to generate a challan.")
                        else:
                            vehicle_id = insert_vehicle(corrected_plate_video, owner_name_video, vehicle_model_video)
                            if all_violations_set:
                                total_fine = challan_gen.generate(corrected_plate_video, list(all_violations_set), original_file_path)
                                st.success(f"E-Challan Generated for {corrected_plate_video}. Total Fine: ₹{total_fine}")
                            else:
                                st.info("No violations detected in video. No challan generated.")
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.exception(e) # Display full traceback for debugging
        finally:
            # Clean up the temporary directory
            if temp_dir:
                clear_temp_directory(temp_dir)
            if 'detected_plate_video' in st.session_state:
                del st.session_state.detected_plate_video


elif menu == "Vehicle Info":
    st.subheader("View Vehicle Details and Violations")
    plate_input = st.text_input("Enter Vehicle Number:")
    if st.button("Fetch Information"):
        if plate_input:
            vehicle_details, violation_records = get_vehicle_info(plate_input.upper()) # Ensure consistent casing
            if vehicle_details:
                st.write("### Vehicle Details:")
                st.write(f"**Registration No:** {vehicle_details[0]}")
                st.write(f"**Owner Name:** {vehicle_details[1] or 'N/A'}")
                st.write(f"**Vehicle Model:** {vehicle_details[2] or 'N/A'}")

                if violation_records:
                    st.write("### Violation History:")
                    df_violations = pd.DataFrame(violation_records, columns=["Violation Type", "Fine Amount (₹)", "Timestamp"])
                    st.dataframe(df_violations)
                    total_fine = df_violations["Fine Amount (₹)"].sum()
                    st.write(f"**Total Fine Due/Paid:** ₹{total_fine}")
                else:
                    st.info("No violation history found for this vehicle.")
            else:
                st.warning("Vehicle not found in the database. Please check the number or register it.")
        else:
            st.warning("Please enter a vehicle number to search.")

elif menu == "Analytics":
    st.subheader("Violation Analytics")
    df_all_violations = execute_query("SELECT * FROM violations") # Get all violations using your manager function

    if not df_all_violations.empty:
        # Convert timestamp to datetime objects for time-based analysis
        df_all_violations['timestamp'] = pd.to_datetime(df_all_violations['timestamp'])

        st.write("### Overall Violation Distribution")
        fig_pie, ax_pie = plt.subplots(figsize=(8, 6))
        violation_counts = df_all_violations['violation_type'].value_counts()
        if not violation_counts.empty:
            ax_pie.pie(violation_counts, labels=violation_counts.index, autopct='%1.1f%%', startangle=90)
            ax_pie.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig_pie)
        else:
            st.info("No violation data to display pie chart.")


        st.write("### Violations Over Time")
        # Group by date and count violations
        df_all_violations['date'] = df_all_violations['timestamp'].dt.date
        violations_per_day = df_all_violations.groupby('date').size().reset_index(name='count')
        fig_line, ax_line = plt.subplots(figsize=(10, 5))
        sns.lineplot(x='date', y='count', data=violations_per_day, marker='o', ax=ax_line)
        ax_line.set_title('Number of Violations Over Time')
        ax_line.set_xlabel('Date')
        ax_line.set_ylabel('Number of Violations')
        ax_line.tick_params(axis='x', rotation=45)
        st.pyplot(fig_line)


        st.write("### Total Fines by Violation Type")
        fines_by_type = df_all_violations.groupby('violation_type')['fine_amount'].sum().sort_values(ascending=False)
        fig_bar_fines, ax_bar_fines = plt.subplots(figsize=(8, 5))
        sns.barplot(x=fines_by_type.index, y=fines_by_type.values, ax=ax_bar_fines)
        ax_bar_fines.set_title('Total Fines Collected by Violation Type')
        ax_bar_fines.set_xlabel('Violation Type')
        ax_bar_fines.set_ylabel('Total Fine Amount (₹)')
        ax_bar_fines.tick_params(axis='x', rotation=45)
        st.pyplot(fig_bar_fines)

    else:
        st.info("No violation data available for analytics yet.")

elif menu == "SQL Query":
    st.subheader("Execute Custom SQL Query")
    st.warning("🚨 **Caution:** This feature allows direct interaction with the database. Only use `SELECT` queries to avoid unintended data modifications. Incorrect queries may cause errors.")
    query = st.text_area("Enter your SQL query here:", height=150)

    if st.button("Execute Query"):
        if query.strip():
            if query.lower().startswith("select"):
                try:
                    result_df = execute_query(query)
                    if not result_df.empty:
                        st.write("### Query Results:")
                        st.dataframe(result_df)
                    else:
                        st.info("Query executed successfully, but returned no results.")
                except Exception as e:
                    st.error(f"Error executing query: {e}")
                    st.exception(e) # Display full traceback for debugging
            else:
                st.error("Only `SELECT` queries are allowed for safety reasons.")
        else:
            st.warning("Please enter an SQL query to execute.")
