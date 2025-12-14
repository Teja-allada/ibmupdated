from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
from sort_enhanced import SortEnhanced
from datetime import datetime, timedelta
import json
import time
from collections import defaultdict, deque
import os
from calibration import AutoCalibrator
from data_export import DataExporter

class TrafficMonitor:
    def __init__(self, video_source, model_path='yolo11n.pt', class_file='coco.names', fps_override=None):
        self.load_classes(class_file)
        self.model = YOLO(model_path)
        self.tracker = SortEnhanced(max_age=30, min_hits=3, iou_threshold=0.3)
        self.cap = cv2.VideoCapture(video_source)

        # Video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        if fps_override:
            self.fps = fps_override
        print(f"Using FPS: {self.fps}")
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.total_count = []
        self.vehicle_info = {}
        self.class_counts = defaultdict(int)
        self.hourly_counts = defaultdict(lambda: defaultdict(int))
        self.track_speed_buffer = defaultdict(lambda: deque(maxlen=7))
        self.prev_positions = {}
        self.prev_speed = {}
        self.prev_ground_positions = {}
        self.prev_ts = {}
        # Speed smoothing/limits to prevent spikes
        self.speed_kmh_max = 80      # cap displayed speed
        self.max_px_step = 50         # cap per-frame pixel displacement used for fallback
        # Acceleration sanity limit (~7 m/s^2 typical)
        self.accel_max_mps2 = 7.0
        # Perspective correction (top of frame has less pixels per meter)
        self.perspective_enabled = True
        self.perspective_min = 0.6    # scaling at top of frame (0.3–1.0)
        
        # Position history for windowed speed and stationarity detection
        # History buffers and thresholds
        self.position_history = defaultdict(lambda: deque(maxlen=15))
        self.px_stationary_thresh = 1.0  # legacy fallback
        self.speed_deadband_kmh = 3.0    # speeds below this are treated as 0
        # Perspective-aware stationarity: treat below ~2 km/h as stationary
        self.speed_stationary_kmh = 2.0
        self.stationary_min_samples = 7
        self.conf_threshold = 0.3
        self.lane_coordinates = self.calculate_lane_coordinates()
        self.calibrator = AutoCalibrator()
        self.is_calibrated = False
        self.pixels_per_meter = 30  # Default fallback
        # Dual-line timing setup (robust speed via D/T)
        self.line_x1 = int(self.frame_width * 0.1)
        self.line_x2 = int(self.frame_width * 0.9)
        self.line_y1 = int(self.frame_height * 0.55)
        self.line_y2 = int(self.frame_height * 0.65)
        self.line_distance_meters = 10.0  # known separation between lines in meters (tunable)
        self.cross_times = defaultdict(lambda: {'first': None, 'second': None})
        
        self.data_exporter = DataExporter()
        self.frame_count = 0
        self.processing_times = []
        self.start_time = time.time()

        # Recording setup
        self.setup_video_writer()

        print("🚗 Enhanced Vehicle Recognition System Initialized")
        print(f"📹 Video Resolution: {self.frame_width}x{self.frame_height}")
        print(f"⚡ FPS: {self.fps}")

    def load_classes(self, class_file):
        with open(class_file, "r") as f:
            self.classNames = [line.strip() for line in f.readlines()]
        self.vehicle_classes = ["car", "truck", "bus", "motorbike"]

    def calculate_lane_coordinates(self):
        return [
            int(self.frame_width * 0.1),
            int(self.frame_height * 0.6),
            int(self.frame_width * 0.9),
            int(self.frame_height * 0.6)
        ]

    def setup_video_writer(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f'enhanced_traffic_recording_{timestamp}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, self.fps,
                                            (self.frame_width, self.frame_height))
        print(f"📽️ Recording to: {output_path}")

    def auto_calibrate(self, detections, frame):
        if not self.is_calibrated and len(detections) > 0:
            car_detections = []
            for detection in detections:
                if detection['class'] == 'car':
                    if detection['width'] > detection['height'] and detection['width'] > 40:
                        car_detections.append(detection)
            if len(car_detections) >= 3:
                calibration_result = self.calibrator.calibrate_from_vehicles(car_detections)
                if calibration_result:
                    self.pixels_per_meter = calibration_result
                    self.is_calibrated = True
                    print(f"🎯 Auto-calibration successful: {self.pixels_per_meter:.2f} pixels/meter")
                    cv2.putText(frame, "CALIBRATED", (self.frame_width - 150, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def pixels_per_meter_at_y(self, y):
        if not self.perspective_enabled:
            return max(1, self.pixels_per_meter)
        # Linear scaling across vertical axis: top uses perspective_min, bottom uses 1.0
        y_norm = float(y) / float(max(1, self.frame_height)) if y is not None else 1.0
        scale = self.perspective_min + (1.0 - self.perspective_min) * y_norm
        return max(1, self.pixels_per_meter * scale)
    
    def calculate_speed_from_velocity(self, velocity_pixels_per_frame, y=None):
        base_ppm = self.pixels_per_meter_at_y(y)
        if base_ppm <= 1:
            return 0.0
        velocity_mps = (velocity_pixels_per_frame / base_ppm) * self.fps
        speed_kmh = velocity_mps * 3.6
        return float(np.clip(speed_kmh, 0, 200))

    def process_frame(self, frame):
        frame_start_time = time.time()
        self.frame_count += 1
        # Undistort the frame if intrinsics are available
        frame = self.calibrator.undistort(frame)
        
        results = self.model(frame, stream=True)
        detections, detection_data = [], []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0])
                if (self.classNames[cls] in self.vehicle_classes and
                    conf > self.conf_threshold):
                    width, height = x2 - x1, y2 - y1
                    if width > 30 and height > 30:
                        detection = np.array([x1, y1, x2, y2, conf])
                        detections.append(detection)
                        detection_data.append({
                            'bbox': [x1, y1, x2, y2],
                            'class': self.classNames[cls],
                            'confidence': conf,
                            'width': width,
                            'height': height
                        })

        self.auto_calibrate(detection_data, frame)
        tracked_objects = self.tracker.update(np.array(detections) if len(detections) else np.empty((0, 5)))

        # Draw virtual lines (two green lines for timing, one red legacy line)
        cv2.line(frame, (self.line_x1, self.line_y1), (self.line_x2, self.line_y1), (0, 255, 0), 3)
        cv2.line(frame, (self.line_x1, self.line_y2), (self.line_x2, self.line_y2), (0, 255, 0), 3)
        cv2.line(frame, (self.lane_coordinates[0], self.lane_coordinates[1]),
                 (self.lane_coordinates[2], self.lane_coordinates[3]), (0, 0, 255), 2)

        current_hour = datetime.now().strftime("%H")
        current_time = datetime.now()

        for tracked_obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, tracked_obj[:5])
            # Track bottom-center point to approximate road contact
            cx, cy = (x1 + x2) // 2, y2
            # Update position history for stationarity detection
            if obj_id in self.prev_positions:
                dx = cx - self.prev_positions[obj_id][0]
                dy = cy - self.prev_positions[obj_id][1]
                step = float(np.sqrt(dx*dx + dy*dy))
                self.position_history[obj_id].append(step)
            else:
                self.position_history[obj_id].append(0.0)
            
            # Vehicle classification
            if obj_id not in self.vehicle_info:
                min_dist, closest_class = float('inf'), "Unknown"
                for det in detection_data:
                    det_cx = (det['bbox'][0] + det['bbox'][2]) // 2
                    det_cy = (det['bbox'][1] + det['bbox'][3]) // 2
                    dist = np.sqrt((cx - det_cx)**2 + (cy - det_cy)**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_class = det['class']
                self.vehicle_info[obj_id] = closest_class

            # Use tracker-reported velocity magnitude (pixels/frame)
            v_tracker = self.tracker.get_velocity(obj_id) or 0.0
            # Fallback displacement
            if obj_id in self.prev_positions:
                dx = cx - self.prev_positions[obj_id][0]
                dy = cy - self.prev_positions[obj_id][1]
                v_fallback = float(np.sqrt(dx * dx + dy * dy))
            else:
                v_fallback = 0.0
            if v_fallback > self.max_px_step:
                v_fallback = self.max_px_step
            velocity_pixels_per_frame = v_tracker if v_tracker > 1e-3 else v_fallback
            speed_kmh = self.calculate_speed_from_velocity(velocity_pixels_per_frame, cy)
            
            speed_world_kmh = None
            ground_pt = self.calibrator.project_to_ground(cx, cy)
            prev_ts_obj = self.prev_ts.get(obj_id)
            if ground_pt is not None and obj_id in self.prev_ground_positions and prev_ts_obj is not None:
                dt_obj = (datetime.fromtimestamp(self.start_time + (self.cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0)) - prev_ts_obj).total_seconds()
                if dt_obj > 1e-3:
                    dist_m = self.calibrator.world_distance(self.prev_ground_positions[obj_id], ground_pt)
                    speed_world_kmh = float((dist_m / dt_obj) * 3.6)
            base_speed_kmh = speed_world_kmh if speed_world_kmh is not None else speed_kmh
            
            samples = len(self.position_history[obj_id])
            if samples >= self.stationary_min_samples:
                ppm = self.pixels_per_meter_at_y(cy)
                px_thresh = float((ppm * (self.speed_stationary_kmh / 3.6)) / max(1e-3, float(self.fps)))
                median_step = float(np.median(self.position_history[obj_id]))
                if (median_step <= px_thresh) and (velocity_pixels_per_frame <= px_thresh * 1.25) and (speed_world_kmh is None or speed_world_kmh <= self.speed_stationary_kmh):
                    base_speed_kmh = 0.0
                    self.track_speed_buffer[obj_id].clear()
                    self.prev_speed[obj_id] = 0.0

            self.track_speed_buffer[obj_id].append(base_speed_kmh)
            smoothed_speed = np.median(self.track_speed_buffer[obj_id]) if len(self.track_speed_buffer[obj_id]) >= 3 else base_speed_kmh

            prev = self.prev_speed.get(obj_id, smoothed_speed)
            dt = 1.0 / max(1e-3, float(self.fps))
            dv_max_kmh = float(self.accel_max_mps2 * dt * 3.6)
            delta = float(smoothed_speed - prev)
            limited_smoothed = prev + float(np.clip(delta, -dv_max_kmh, dv_max_kmh))
            final_speed = float(min(limited_smoothed, self.speed_kmh_max))

            if final_speed <= self.speed_deadband_kmh:
                ppm = self.pixels_per_meter_at_y(cy)
                px_thresh_db = float((ppm * (self.speed_stationary_kmh / 3.6)) / max(1e-3, float(self.fps)))
                if len(self.position_history[obj_id]) >= 3 and np.median(self.position_history[obj_id]) <= px_thresh_db:
                    final_speed = 0.0

            prev_y = self.prev_positions[obj_id][1] if obj_id in self.prev_positions else None
            frame_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            frame_ts = datetime.fromtimestamp(self.start_time + (frame_msec/1000.0)) if frame_msec > 0 else datetime.now()

            if prev_y is not None:
                crossed_first = (prev_y < self.line_y1 and cy >= self.line_y1)
                crossed_second = (prev_y < self.line_y2 and cy >= self.line_y2)
                prev_ts_obj = self.prev_ts.get(obj_id, frame_ts)
                if crossed_first and (self.cross_times[obj_id]['first'] is None):
                    if cy != prev_y:
                        alpha1 = float((self.line_y1 - prev_y) / (cy - prev_y))
                        dt_s = (frame_ts - prev_ts_obj).total_seconds()
                        t_first = prev_ts_obj + timedelta(seconds=max(0.0, min(1.0, alpha1)) * dt_s)
                        self.cross_times[obj_id]['first'] = t_first
                    else:
                        self.cross_times[obj_id]['first'] = frame_ts
                if crossed_second and (self.cross_times[obj_id]['second'] is None):
                    if cy != prev_y:
                        alpha2 = float((self.line_y2 - prev_y) / (cy - prev_y))
                        dt_s2 = (frame_ts - prev_ts_obj).total_seconds()
                        t_second = prev_ts_obj + timedelta(seconds=max(0.0, min(1.0, alpha2)) * dt_s2)
                        self.cross_times[obj_id]['second'] = t_second
                    else:
                        self.cross_times[obj_id]['second'] = frame_ts
                t1 = self.cross_times[obj_id]['first']
                t2 = self.cross_times[obj_id]['second']
                if t1 and t2:
                    dt_cross = (t2 - t1).total_seconds()
                    dt_min = float(self.line_distance_meters / max(1e-6, (self.speed_kmh_max / 3.6)))
                    if dt_cross >= dt_min:
                        speed_cross_kmh = float((self.line_distance_meters / dt_cross) * 3.6)
                        final_speed = float(min(max(final_speed, speed_cross_kmh), self.speed_kmh_max))
            
            self.prev_speed[obj_id] = final_speed
            self.prev_positions[obj_id] = (cx, cy)
            if ground_pt is not None:
                self.prev_ground_positions[obj_id] = ground_pt
            self.prev_ts[obj_id] = frame_ts

            self.draw_enhanced_vehicle(frame, x1, y1, x2, y2, obj_id, final_speed)

            # Crossing and data export
            if (self.lane_coordinates[0] < cx < self.lane_coordinates[2] and abs(cy - self.lane_coordinates[1]) < 15):
                if obj_id not in self.total_count:
                    self.total_count.append(obj_id)
                    self.class_counts[self.vehicle_info[obj_id]] += 1
                    self.hourly_counts[current_hour][self.vehicle_info[obj_id]] += 1
                    # Export a single record for this crossing event
                    self.data_exporter.add_record({
                        'timestamp': frame_ts.isoformat(),
                        'vehicle_id': obj_id,
                        'vehicle_class': self.vehicle_info[obj_id],
                        'speed_kmh': round(self.prev_speed.get(obj_id, final_speed), 2),
                        'position_x': cx,
                        'position_y': cy,
                        'frame_number': self.frame_count,
                        'calibration_ppm': round(self.pixels_per_meter_at_y(cy), 2) if self.is_calibrated else 0
                    })

        # Analytics overlay and performance
        self.draw_enhanced_analytics(frame)
        processing_time = time.time() - frame_start_time
        self.processing_times.append(processing_time)
        self.video_writer.write(frame)
        return frame

    def draw_enhanced_vehicle(self, frame, x1, y1, x2, y2, obj_id, speed_kmh):
        vehicle_class = self.vehicle_info.get(obj_id, "Unknown")
        colors = {
            'car': (0, 255, 0), 'truck': (0, 0, 255),
            'bus': (255, 0, 0), 'motorbike': (255, 255, 0)}
        color = colors.get(vehicle_class, (255, 255, 255))
        cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), l=12, rt=3, colorR=color)
        id_text = f"ID:{obj_id} {vehicle_class.upper()}"
        cvzone.putTextRect(frame, id_text, (x1, max(35, y1)), scale=0.8, thickness=2, colorR=color)
        speed_text = f"{int(speed_kmh)} km/h"
        speed_bg_color = (0, 255, 0) if speed_kmh <= 50 else (0, 165, 255) if speed_kmh <= 80 else (0, 0, 255)
        speed_y = max(20, y1 - 10)
        cvzone.putTextRect(frame, speed_text, (x1, speed_y), scale=1.2, thickness=3, colorR=speed_bg_color)
        if vehicle_class == 'car' and not self.is_calibrated:
            cv2.circle(frame, (x1 + 10, y1 + 10), 5, (0, 255, 255), -1)

    def draw_enhanced_analytics(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 250), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, "ENHANCED TRAFFIC MONITOR", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset = 60
        cv2.putText(frame, f"Total Vehicles: {len(self.total_count)}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for vehicle_class in self.vehicle_classes:
            y_offset += 25
            count = self.class_counts[vehicle_class]
            cv2.putText(frame, f"{vehicle_class.title()}: {count}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        y_offset += 30
        calib_text = "CALIBRATED" if self.is_calibrated else "AUTO-CALIBRATING..."
        calib_color = (0, 255, 0) if self.is_calibrated else (0, 255, 255)
        cv2.putText(frame, calib_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, calib_color, 2)
        if self.is_calibrated:
            y_offset += 20
            cv2.putText(frame, f"Pixels/meter: {self.pixels_per_meter:.1f}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 25
        if self.processing_times:
            avg_time = sum(self.processing_times[-30:]) / len(self.processing_times) if self.processing_times else 0
            fps_actual = 1.0 / avg_time if avg_time > 0 else 0
            cv2.putText(frame, f"Processing FPS: {fps_actual:.1f}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def save_final_analytics(self):
        csv_path = self.data_exporter.save_to_csv()
        analytics_data = {
            'session_info': {
                'total_vehicles': len(self.total_count),
                'total_frames': self.frame_count,
                'total_runtime': time.time() - self.start_time,
                'calibrated': self.is_calibrated,
                'pixels_per_meter': self.pixels_per_meter
            },
            'vehicle_distribution': dict(self.class_counts),
            'hourly_counts': dict(self.hourly_counts),
            'performance_stats': {
                'avg_processing_time': sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0,
                'avg_fps': len(self.processing_times) / sum(self.processing_times) if self.processing_times else 0
            }
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = f'enhanced_traffic_analytics_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(analytics_data, f, indent=4)
        print(f"\n📊 Analytics saved:")
        print(f"   📈 CSV: {csv_path}")
        print(f"   📋 JSON: {json_path}")

    def run(self):
        print("\n🚀 Starting Enhanced Vehicle Recognition...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save current analytics")
        print("  'c' - Force calibration reset")
        try:
            while True:
                success, frame = self.cap.read()
                if not success:
                    print("📹 End of video stream")
                    break
                processed_frame = self.process_frame(frame)
                self.video_writer.write(processed_frame)
                cv2.imshow("Enhanced Traffic Monitoring System", processed_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n🛑 Stopping system...")
                    break
                elif key == ord('s'):
                    print("\n💾 Saving current analytics...")
                    self.save_final_analytics()
                elif key == ord('c'):
                    print("\n🔄 Resetting calibration...")
                    self.is_calibrated = False
                    self.pixels_per_meter = 30
        finally:
            print("\n🏁 Session complete!")
            self.save_final_analytics()
            self.cap.release()
            self.video_writer.release()
            cv2.destroyAllWindows()
            print(f"\n📈 SESSION SUMMARY:")
            print(f"   🚗 Total vehicles detected: {len(self.total_count)}")
            print(f"   ⏱️  Total runtime: {time.time() - self.start_time:.1f} seconds")
            print(f"   🎯 Calibration: {'✅ Auto-calibrated' if self.is_calibrated else '❌ Default values used'}")
            if self.is_calibrated:
                print(f"   📏 Pixels per meter: {self.pixels_per_meter:.2f}")

if __name__ == "__main__":
    monitor = TrafficMonitor(
        video_source="video_10122025.mp4",
        model_path="yolo11n.pt",
        class_file="coco.names",
        fps_override=30    # Set to your real video FPS!
    )
    monitor.run()
