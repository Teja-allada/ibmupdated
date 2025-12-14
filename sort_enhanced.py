
from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

np.random.seed(0)


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
             + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return(o)


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
    """
    Enhanced Kalman tracker that provides velocity extraction capabilities
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4) 

        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],  # x = x + vx*dt
            [0,1,0,0,0,1,0],  # y = y + vy*dt  
            [0,0,1,0,0,0,1],  # s = s + vs*dt
            [0,0,0,1,0,0,0],  # r = r (aspect ratio constant)
            [0,0,0,0,1,0,0],  # vx = vx (constant velocity)
            [0,0,0,0,0,1,0],  # vy = vy (constant velocity)
            [0,0,0,0,0,0,1]   # vs = vs (scale velocity)
        ])

        # Measurement function (we observe x, y, s, r)
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])

        # Measurement noise covariance
        self.kf.R[2:,2:] *= 10.

        # Initial covariance (high uncertainty for unobservable velocities)
        self.kf.P[4:,4:] *= 1000. 
        self.kf.P *= 10.

        # Process noise covariance
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        # Initialize state with bbox
        self.kf.x[:4] = convert_bbox_to_z(bbox)

        # Tracking metadata
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        # Store velocity history for smoothing
        self.velocity_history = []
        self.max_velocity_history = 10

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)

    def get_velocity(self):
        """
        Extract velocity from Kalman filter state
        Returns velocity in pixels per frame as magnitude
        """
        vx = self.kf.x[4, 0]  # x velocity
        vy = self.kf.x[5, 0]  # y velocity

        # Calculate velocity magnitude
        velocity_magnitude = np.sqrt(vx**2 + vy**2)

        # Store in history for smoothing
        self.velocity_history.append(velocity_magnitude)
        if len(self.velocity_history) > self.max_velocity_history:
            self.velocity_history.pop(0)

        # Return smoothed velocity (average of recent measurements)
        if len(self.velocity_history) > 0:
            smoothed_velocity = np.mean(self.velocity_history)
            return smoothed_velocity
        else:
            return 0.0

    def get_velocity_vector(self):
        """
        Returns velocity as (vx, vy) vector in pixels per frame
        """
        return (self.kf.x[4, 0], self.kf.x[5, 0])

    def get_position_velocity(self):
        """
        Returns both position (x, y) and velocity (vx, vy)
        """
        x = self.kf.x[0, 0]
        y = self.kf.x[1, 0] 
        vx = self.kf.x[4, 0]
        vy = self.kf.x[5, 0]
        return (x, y, vx, vy)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    # Filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class SortEnhanced(object):
    """
    Enhanced SORT tracker with velocity extraction capabilities
    """
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Initialize enhanced SORT tracker

        Parameters:
        max_age: Maximum number of frames to keep alive a track without detections
        min_hits: Minimum number of associated detections before track is initialized  
        iou_threshold: Minimum IOU for match
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

        print(f"🔧 Enhanced SORT initialized (max_age={max_age}, min_hits={min_hits}, iou={iou_threshold})")

    def update(self, dets=np.empty((0, 5))):
        """
        Enhanced update method that maintains tracking state

        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections
        Returns: a similar array, where the last column is the object ID

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1

        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # Create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
            i -= 1
            # Remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)

        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0,5))

    def get_velocity(self, track_id):
        """
        Get velocity for a specific track ID

        Parameters:
        track_id: The track ID to get velocity for

        Returns:
        velocity magnitude in pixels per frame, or None if track not found
        """
        # Convert track_id to internal tracker index (subtract 1 as we add 1 in update)
        internal_id = track_id - 1

        for tracker in self.trackers:
            if tracker.id == internal_id:
                return tracker.get_velocity()
        return None

    def get_all_velocities(self):
        """
        Get velocities for all active tracks

        Returns:
        Dictionary mapping track_id to velocity
        """
        velocities = {}
        for tracker in self.trackers:
            track_id = tracker.id + 1  # Convert to external ID
            velocities[track_id] = tracker.get_velocity()
        return velocities

    def get_tracker_info(self, track_id):
        """
        Get comprehensive info for a track

        Returns:
        Dictionary with position, velocity, age, hits, etc.
        """
        internal_id = track_id - 1

        for tracker in self.trackers:
            if tracker.id == internal_id:
                x, y, vx, vy = tracker.get_position_velocity()
                return {
                    'track_id': track_id,
                    'position': (x, y),
                    'velocity': (vx, vy),
                    'velocity_magnitude': tracker.get_velocity(),
                    'age': tracker.age,
                    'hits': tracker.hits,
                    'time_since_update': tracker.time_since_update,
                    'hit_streak': tracker.hit_streak
                }
        return None

    def get_active_track_count(self):
        """Get number of currently active tracks"""
        return len(self.trackers)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Enhanced SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=1)
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # Test code for Enhanced SORT
    args = parse_args()
    display = args.display
    phase = args.phase
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3) # used only for display

    if(display):
        if not os.path.exists('mot_benchmark'):
            print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
            exit()
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect='equal')

    if not os.path.exists('output'):
        os.makedirs('output')

    pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
    for seq_dets_fn in glob.glob(pattern):
        mot_tracker = SortEnhanced(max_age=args.max_age,
                                   min_hits=args.min_hits,
                                   iou_threshold=args.iou_threshold)
        seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
        seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]

        with open(os.path.join('output', '%s.txt'%(seq)),'w') as out_file:
            print("Processing %s."%(seq))
            for frame in range(int(seq_dets[:,0].max())):
                frame += 1 # detection and frame numbers begin at 1
                dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
                dets[:, 2:4] += dets[:, 0:2] # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                total_frames += 1

                if(display):
                    fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg'%(frame))
                    im =io.imread(fn)
                    ax1.imshow(im)
                    plt.title(seq + ' Tracked Targets')

                start_time = time.time()
                trackers = mot_tracker.update(dets)
                cycle_time = time.time() - start_time
                total_time += cycle_time

                for d in trackers:
                    # Get velocity for this tracker
                    track_id = int(d[4])
                    velocity = mot_tracker.get_velocity(track_id)

                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1,%.2f'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1],velocity),file=out_file)
                    if(display):
                        d = d.astype(np.int32)
                        ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))

                if(display):
                    fig.canvas.flush_events()
                    plt.draw()
                    ax1.cla()

    if total_frames > 0:
        fps = total_frames / total_time
    else:
        fps = 0.0

    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, fps))

    if(display):
        print("Note: to get real runtime results run without the option: --display")


