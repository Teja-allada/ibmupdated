
import pandas as pd
import numpy as np
import csv
from datetime import datetime
import os
from collections import deque

class DataExporter:
    """
    Enhanced data export system for vehicle tracking data
    """

    def __init__(self, buffer_size=1000):
        self.records = deque(maxlen=buffer_size)
        self.csv_headers = [
            'timestamp',
            'vehicle_id', 
            'vehicle_class',
            'speed_kmh',
            'position_x',
            'position_y',
            'frame_number',
            'calibration_ppm'
        ]

        # Session info
        self.session_start = datetime.now()
        self.export_count = 0

        print("📊 DataExporter initialized")

    def add_record(self, record_data):
        """
        Add a new vehicle record to the buffer

        Expected record format:
        {
            'timestamp': ISO timestamp string,
            'vehicle_id': int,
            'vehicle_class': str,
            'speed_kmh': float,
            'position_x': int,
            'position_y': int, 
            'frame_number': int,
            'calibration_ppm': float
        }
        """
        # Validate required fields
        required_fields = ['timestamp', 'vehicle_id', 'vehicle_class', 'speed_kmh']
        for field in required_fields:
            if field not in record_data:
                print(f"⚠️  Warning: Missing required field '{field}' in record")
                return False

        # Add to buffer
        self.records.append(record_data.copy())

        # Auto-save periodically
        if len(self.records) % 100 == 0:
            print(f"📈 Buffer: {len(self.records)} records")

        return True

    def save_to_csv(self, filename=None, include_all_data=True):
        """
        Save all records to CSV file
        """
        if not self.records:
            print("⚠️  No records to save")
            return None

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'enhanced_vehicle_data_{timestamp}.csv'

        try:
            # Convert to pandas DataFrame for easy manipulation
            df = pd.DataFrame(list(self.records))

            # Add computed columns
            df = self.add_computed_columns(df)

            # Sort by timestamp
            df = df.sort_values('timestamp')

            # Save to CSV
            df.to_csv(filename, index=False)

            self.export_count += 1

            print(f"✅ Exported {len(df)} records to {filename}")

            # Print summary statistics
            self.print_export_summary(df)

            return filename

        except Exception as e:
            print(f"❌ Error saving CSV: {e}")
            return None

    def add_computed_columns(self, df):
        """
        Add computed columns for enhanced analysis
        """
        # Convert timestamp to datetime for calculations
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Add time-based columns
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['second'] = df['timestamp'].dt.second

        # Add speed categories
        df['speed_category'] = df['speed_kmh'].apply(self.categorize_speed)

        # Add vehicle size category
        df['size_category'] = df['vehicle_class'].apply(self.categorize_vehicle_size)

        # Calculate time since session start
        session_start_ts = pd.to_datetime(self.session_start)
        df['seconds_since_start'] = (df['timestamp'] - session_start_ts).dt.total_seconds()

        # Add sequential numbering
        df['detection_sequence'] = range(1, len(df) + 1)

        return df

    def categorize_speed(self, speed_kmh):
        """
        Categorize speed into ranges
        """
        if speed_kmh < 20:
            return 'slow'
        elif speed_kmh < 50:
            return 'moderate'
        elif speed_kmh < 80:
            return 'fast'
        else:
            return 'very_fast'

    def categorize_vehicle_size(self, vehicle_class):
        """
        Categorize vehicle by size
        """
        size_mapping = {
            'motorbike': 'small',
            'car': 'medium', 
            'truck': 'large',
            'bus': 'large'
        }
        return size_mapping.get(vehicle_class.lower(), 'unknown')

    def print_export_summary(self, df):
        """
        Print summary statistics of exported data
        """
        print("\n📊 Export Summary:")
        print(f"   Total records: {len(df)}")
        print(f"   Time span: {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Vehicle class distribution
        print("\n🚗 Vehicle Distribution:")
        class_counts = df['vehicle_class'].value_counts()
        for vehicle_class, count in class_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {vehicle_class}: {count} ({percentage:.1f}%)")

        # Speed statistics
        print("\n🏃 Speed Statistics:")
        print(f"   Average: {df['speed_kmh'].mean():.1f} km/h")
        print(f"   Median: {df['speed_kmh'].median():.1f} km/h")
        print(f"   Min: {df['speed_kmh'].min():.1f} km/h")
        print(f"   Max: {df['speed_kmh'].max():.1f} km/h")

        # Speed categories
        speed_cats = df['speed_category'].value_counts()
        print("\n⚡ Speed Categories:")
        for category, count in speed_cats.items():
            percentage = (count / len(df)) * 100
            print(f"   {category}: {count} ({percentage:.1f}%)")

    def save_hourly_summary(self, filename=None):
        """
        Save hourly traffic summary
        """
        if not self.records:
            return None

        df = pd.DataFrame(list(self.records))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour

        # Create hourly summary
        hourly_stats = df.groupby(['hour', 'vehicle_class']).agg({
            'vehicle_id': 'count',
            'speed_kmh': ['mean', 'std']
        }).round(2)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'hourly_traffic_summary_{timestamp}.csv'

        hourly_stats.to_csv(filename)
        print(f"📅 Hourly summary saved to {filename}")
        return filename

    def get_real_time_stats(self):
        """
        Get current statistics for real-time display
        """
        if not self.records:
            return {}

        # Get recent records (last 100)
        recent_records = list(self.records)[-100:]
        df = pd.DataFrame(recent_records)

        if df.empty:
            return {}

        stats = {
            'total_vehicles': len(df),
            'avg_speed': df['speed_kmh'].mean(),
            'vehicle_distribution': df['vehicle_class'].value_counts().to_dict(),
            'speed_distribution': {
                'slow': len(df[df['speed_kmh'] < 20]),
                'moderate': len(df[(df['speed_kmh'] >= 20) & (df['speed_kmh'] < 50)]),
                'fast': len(df[(df['speed_kmh'] >= 50) & (df['speed_kmh'] < 80)]),
                'very_fast': len(df[df['speed_kmh'] >= 80])
            }
        }

        return stats

    def export_for_analysis(self, analysis_type="anomaly_detection"):
        """
        Export data in format suitable for specific analysis
        """
        if not self.records:
            return None

        df = pd.DataFrame(list(self.records))
        df = self.add_computed_columns(df)

        if analysis_type == "anomaly_detection":
            # Export features relevant for anomaly detection
            features = [
                'vehicle_id', 'vehicle_class', 'speed_kmh', 
                'position_x', 'position_y', 'hour', 'minute',
                'speed_category', 'size_category', 'seconds_since_start'
            ]
            analysis_df = df[features].copy()

            # Add more features for anomaly detection
            analysis_df['speed_deviation'] = abs(analysis_df['speed_kmh'] - analysis_df['speed_kmh'].mean())
            analysis_df['position_distance'] = np.sqrt(analysis_df['position_x']**2 + analysis_df['position_y']**2)

        elif analysis_type == "traffic_flow":
            # Features for traffic flow analysis
            features = [
                'timestamp', 'vehicle_class', 'speed_kmh',
                'hour', 'minute', 'speed_category'
            ]
            analysis_df = df[features].copy()

        else:
            analysis_df = df.copy()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{analysis_type}_data_{timestamp}.csv'
        analysis_df.to_csv(filename, index=False)

        print(f"🔬 Analysis data exported to {filename}")
        return filename

    def clear_buffer(self):
        """
        Clear the record buffer
        """
        count = len(self.records)
        self.records.clear()
        print(f"🗑️  Cleared {count} records from buffer")

    def get_buffer_status(self):
        """
        Get current buffer status
        """
        return {
            'current_size': len(self.records),
            'max_size': self.records.maxlen,
            'usage_percent': (len(self.records) / self.records.maxlen) * 100,
            'export_count': self.export_count
        }
