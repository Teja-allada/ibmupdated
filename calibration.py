
import numpy as np
import cv2
from collections import deque
import statistics

class AutoCalibrator:
    """
    Automatic camera calibration using vehicle dimensions
    """

    def __init__(self):
        # Standard vehicle dimensions (in meters)
        self.vehicle_dimensions = {
            'car': {
                'length': {'avg': 4.5, 'range': (3.5, 5.5)},
                'width': {'avg': 1.8, 'range': (1.5, 2.2)},
                'height': {'avg': 1.5, 'range': (1.3, 1.8)}
            },
            'truck': {
                'length': {'avg': 8.0, 'range': (6.0, 12.0)},
                'width': {'avg': 2.5, 'range': (2.0, 3.0)},
                'height': {'avg': 3.0, 'range': (2.5, 4.0)}
            },
            'bus': {
                'length': {'avg': 12.0, 'range': (10.0, 15.0)},
                'width': {'avg': 2.5, 'range': (2.3, 2.7)},
                'height': {'avg': 3.2, 'range': (2.8, 3.8)}
            },
            'motorbike': {
                'length': {'avg': 2.2, 'range': (1.8, 2.6)},
                'width': {'avg': 0.8, 'range': (0.6, 1.0)},
                'height': {'avg': 1.2, 'range': (1.0, 1.4)}
            }
        }

        # Calibration data collection
        self.calibration_samples = deque(maxlen=50)  # Store last 50 samples
        self.min_samples_for_calibration = 10

        # Quality thresholds
        self.confidence_threshold = 0.6
        self.aspect_ratio_tolerance = 0.3

        print("🎯 AutoCalibrator initialized")
        print("📏 Reference dimensions loaded for car, truck, bus, motorbike")
        
        # Intrinsics and distortion (optional, filled via calibrate_intrinsics_from_chessboard)
        self.camera_matrix = None
        self.dist_coeffs = None
        self._undistort_map1 = None
        self._undistort_map2 = None
        self._undistort_size = None
        
        # Homography image→ground (meters)
        self.homography = None

    def estimate_distance_from_height(self, bbox_height, real_height, focal_length_approx=800):
        """
        Estimate distance to object using pinhole camera model
        """
        # Approximate distance = (real_height * focal_length) / image_height
        distance = (real_height * focal_length_approx) / bbox_height
        return distance

    def is_valid_car_detection(self, detection):
        """
        Validate if detection is suitable for calibration
        """
        bbox = detection['bbox']
        width_pixels = bbox[2] - bbox[0]
        height_pixels = bbox[3] - bbox[1]

        # Check minimum size
        if width_pixels < 50 or height_pixels < 30:
            return False, "Too small"

        # Check aspect ratio (cars are typically wider than tall when viewed from side)
        aspect_ratio = width_pixels / height_pixels
        if aspect_ratio < 1.0 or aspect_ratio > 4.0:
            return False, f"Bad aspect ratio: {aspect_ratio:.2f}"

        # Check confidence
        if detection['confidence'] < self.confidence_threshold:
            return False, f"Low confidence: {detection['confidence']:.2f}"

        return True, "Valid"

    def estimate_pixels_per_meter_from_width(self, detection):
        """
        Estimate pixels per meter using car width assumption
        """
        if detection['class'] != 'car':
            return None

        is_valid, reason = self.is_valid_car_detection(detection)
        if not is_valid:
            return None

        # Get bounding box dimensions
        bbox = detection['bbox']
        width_pixels = bbox[2] - bbox[0]

        # Assume average car width
        car_width_meters = self.vehicle_dimensions['car']['width']['avg']

        # Calculate pixels per meter
        pixels_per_meter = width_pixels / car_width_meters

        return pixels_per_meter

    def estimate_pixels_per_meter_from_height(self, detection):
        """
        Estimate pixels per meter using vehicle height
        """
        vehicle_class = detection['class']
        if vehicle_class not in self.vehicle_dimensions:
            return None

        bbox = detection['bbox']
        height_pixels = bbox[3] - bbox[1]

        # Get average height for this vehicle type
        avg_height_meters = self.vehicle_dimensions[vehicle_class]['height']['avg']

        # Calculate pixels per meter
        pixels_per_meter = height_pixels / avg_height_meters

        return pixels_per_meter

    def multi_vehicle_calibration(self, detections):
        """
        Use multiple vehicles for more robust calibration
        """
        calibration_estimates = []

        for detection in detections:
            vehicle_class = detection['class']

            if vehicle_class not in self.vehicle_dimensions:
                continue

            # Try width-based estimation (most reliable for cars)
            if vehicle_class == 'car':
                width_estimate = self.estimate_pixels_per_meter_from_width(detection)
                if width_estimate and 10 < width_estimate < 200:  # Reasonable range
                    calibration_estimates.append({
                        'method': 'car_width',
                        'pixels_per_meter': width_estimate,
                        'confidence': detection['confidence'],
                        'vehicle_class': vehicle_class
                    })

            # Try height-based estimation
            height_estimate = self.estimate_pixels_per_meter_from_height(detection)
            if height_estimate and 5 < height_estimate < 100:  # Reasonable range
                calibration_estimates.append({
                    'method': 'height',
                    'pixels_per_meter': height_estimate,
                    'confidence': detection['confidence'],
                    'vehicle_class': vehicle_class
                })

        return calibration_estimates

    def filter_outliers(self, estimates):
        """
        Remove outlier estimates using statistical methods
        """
        if len(estimates) < 3:
            return estimates

        values = [est['pixels_per_meter'] for est in estimates]

        # Calculate IQR
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1

        # Define outlier bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Filter estimates
        filtered = []
        for est in estimates:
            if lower_bound <= est['pixels_per_meter'] <= upper_bound:
                filtered.append(est)

        return filtered if len(filtered) > 0 else estimates

    def weighted_average_calibration(self, estimates):
        """
        Calculate weighted average based on confidence and method reliability
        """
        if not estimates:
            return None

        # Method weights (car width is most reliable)
        method_weights = {
            'car_width': 3.0,
            'height': 1.0
        }

        total_weight = 0
        weighted_sum = 0

        for est in estimates:
            # Combined weight = method weight * confidence
            method_weight = method_weights.get(est['method'], 1.0)
            confidence_weight = est['confidence']
            total_weight_this = method_weight * confidence_weight

            weighted_sum += est['pixels_per_meter'] * total_weight_this
            total_weight += total_weight_this

        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return None

    def add_calibration_sample(self, pixels_per_meter, quality_score=1.0):
        """
        Add calibration sample to running average
        """
        self.calibration_samples.append({
            'pixels_per_meter': pixels_per_meter,
            'quality': quality_score,
            'timestamp': np.datetime64('now')
        })

    def get_stable_calibration(self):
        """
        Get stable calibration value from accumulated samples
        """
        if len(self.calibration_samples) < self.min_samples_for_calibration:
            return None

        # Get recent samples
        recent_values = [s['pixels_per_meter'] for s in list(self.calibration_samples)[-20:]]

        # Calculate statistics
        median_value = statistics.median(recent_values)
        std_dev = statistics.stdev(recent_values) if len(recent_values) > 1 else 0

        # Check if values are stable (low standard deviation)
        if std_dev < median_value * 0.15:  # Within 15% variation
            return median_value
        else:
            return None

    def calibrate_from_vehicles(self, detections):
        """
        Main calibration method using multiple vehicle detections
        """
        if len(detections) < 1:
            return None

        # Get calibration estimates from all vehicles
        estimates = self.multi_vehicle_calibration(detections)

        if len(estimates) < 1:
            return None

        # Filter outliers
        filtered_estimates = self.filter_outliers(estimates)

        # Calculate weighted average
        calibration_value = self.weighted_average_calibration(filtered_estimates)

        if calibration_value:
            # Add to sample collection
            quality_score = min(len(filtered_estimates) / 5.0, 1.0)  # More vehicles = better quality
            self.add_calibration_sample(calibration_value, quality_score)

            # Try to get stable calibration
            stable_value = self.get_stable_calibration()

            if stable_value:
                print(f"🎯 Stable calibration achieved: {stable_value:.2f} pixels/meter")
                print(f"   Based on {len(self.calibration_samples)} samples")
                return stable_value
            else:
                print(f"📊 Calibration estimate: {calibration_value:.2f} pixels/meter (collecting more samples...)")
                return calibration_value

        return None

    def validate_calibration(self, pixels_per_meter, test_detections):
        """
        Validate calibration using known vehicle dimensions
        """
        if not test_detections:
            return False, "No test detections"

        validation_errors = []

        for detection in test_detections:
            vehicle_class = detection['class']
            if vehicle_class not in self.vehicle_dimensions:
                continue

            bbox = detection['bbox']
            width_pixels = bbox[2] - bbox[0]
            height_pixels = bbox[3] - bbox[1]

            # Convert to meters
            width_meters = width_pixels / pixels_per_meter
            height_meters = height_pixels / pixels_per_meter

            # Compare with expected dimensions
            expected_width = self.vehicle_dimensions[vehicle_class]['width']
            expected_height = self.vehicle_dimensions[vehicle_class]['height']

            # Check if within expected range
            width_error = 0
            if width_meters < expected_width['range'][0]:
                width_error = expected_width['range'][0] - width_meters
            elif width_meters > expected_width['range'][1]:
                width_error = width_meters - expected_width['range'][1]

            height_error = 0
            if height_meters < expected_height['range'][0]:
                height_error = expected_height['range'][0] - height_meters
            elif height_meters > expected_height['range'][1]:
                height_error = height_meters - expected_height['range'][1]

            total_error = width_error + height_error
            validation_errors.append(total_error)

        if validation_errors:
            avg_error = np.mean(validation_errors)
            # Good calibration should have low average error
            is_valid = avg_error < 0.5  # Less than 0.5m average error
            return is_valid, f"Average error: {avg_error:.2f}m"

        return False, "No valid test cases"

    def reset_calibration(self):
        """
        Reset calibration data
        """
        self.calibration_samples.clear()
        print("🔄 Calibration data reset")

    def get_calibration_stats(self):
        """
        Get statistics about current calibration
        """
        if not self.calibration_samples:
            return {
                'sample_count': 0,
                'status': 'No samples'
            }

        values = [s['pixels_per_meter'] for s in self.calibration_samples]

        return {
            'sample_count': len(self.calibration_samples),
            'mean': np.mean(values),
            'median': np.median(values),
            'std_dev': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'status': 'Stable' if len(self.calibration_samples) >= self.min_samples_for_calibration else 'Collecting'
        }

    def calibrate_intrinsics_from_chessboard(self, image_paths, pattern_size=(9, 6), square_size=0.025):
        """
        Calibrate camera intrinsics (K, D) from a list of chessboard image paths.
        pattern_size: (cols, rows) inner corners; square_size: meters per square.
        """
        objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= float(square_size)
        objpoints = []
        imgpoints = []
        img_size = None
        for p in image_paths:
            img = cv2.imread(p)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_size = (gray.shape[1], gray.shape[0])
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                objpoints.append(objp)
                imgpoints.append(corners2)
        if len(objpoints) < 5:
            print("⚠️ Not enough chessboard detections for calibration")
            return None, None
        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        if not ret:
            print("⚠️ calibrateCamera failed")
            return None, None
        self.camera_matrix = K
        self.dist_coeffs = D
        # Precompute undistort maps for speed
        self._undistort_map1, self._undistort_map2 = cv2.initUndistortRectifyMap(K, D, None, K, img_size, cv2.CV_16SC2)
        self._undistort_size = img_size
        print("✅ Intrinsics calibrated. Image size:", img_size)
        return K, D

    def undistort(self, frame):
        """
        Undistort a frame if intrinsics are available; otherwise return the input frame.
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            return frame
        h, w = frame.shape[:2]
        if self._undistort_map1 is not None and (w, h) == self._undistort_size:
            return cv2.remap(frame, self._undistort_map1, self._undistort_map2, interpolation=cv2.INTER_LINEAR)
        # Fallback: direct undistort
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

    def set_homography(self, image_points, world_points):
        """
        Set homography H using image_points (Nx2) and world_points (Nx2, meters) with RANSAC.
        """
        img_pts = np.asarray(image_points, dtype=np.float32)
        wrd_pts = np.asarray(world_points, dtype=np.float32)
        if img_pts.shape[0] < 4 or wrd_pts.shape[0] < 4:
            print("⚠️ Need at least 4 point pairs for homography")
            return False
        H, mask = cv2.findHomography(img_pts, wrd_pts, method=cv2.RANSAC)
        if H is None:
            print("⚠️ findHomography failed")
            return False
        self.homography = H
        print("✅ Homography set (image→ground)")
        return True

    def project_to_ground(self, x, y):
        """
        Project image point (x,y) to ground plane meters using homography.
        Returns (X, Y) or None if H is unavailable.
        """
        if self.homography is None:
            return None
        p = np.array([x, y, 1.0], dtype=np.float32)
        q = self.homography @ p
        if abs(q[2]) < 1e-6:
            return None
        X = float(q[0] / q[2])
        Y = float(q[1] / q[2])
        return (X, Y)

    def world_distance(self, P1, P2):
        """
        Euclidean distance in meters between two ground points (X,Y) tuples.
        """
        return float(np.hypot(P2[0] - P1[0], P2[1] - P1[1]))
