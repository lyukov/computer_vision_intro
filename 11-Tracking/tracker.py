import numpy as np
import os

from moviepy.editor import VideoFileClip

from detection import detection_cast, extract_detections, draw_detections
from metrics import iou_score


class Tracker:
    """Generate detections and build tracklets."""
    def __init__(self, return_images=True, lookup_tail_size=80, labels=None):
        self.return_images = return_images  # Return images or detections?
        self.frame_index = 0
        self.labels = labels  # Tracker label list
        self.detection_history = []  # Saved detection list
        self.last_detected = {}
        self.tracklet_count = 0  # Counter to enumerate tracklets

        # We will search tracklet at last lookup_tail_size frames
        self.lookup_tail_size = lookup_tail_size

    def new_label(self):
        """Get new unique label."""
        self.tracklet_count += 1
        return self.tracklet_count - 1

    def init_tracklet(self, frame):
        """Get new unique label for every detection at frame and return it."""
        # Write code here
        # Use extract_detections and new_label
        result = list(map(
            lambda det: [self.new_label()] + list(det),
            extract_detections(frame)[:, 1:]
        ))
        return np.array(result)

    @property
    def prev_detections(self):
        """Get detections at last lookup_tail_size frames from detection_history.

        One detection at one id.
        """
        detections = []
        # Write code here
        ids = set()
        for detection in self.detection_history[-self.lookup_tail_size:][::-1]:
            for det in detection:
                if det[0] not in ids:
                    ids.add(det[0])
                    detections.append(det)

        return detection_cast(detections)

    def bind_tracklet(self, detections):
        """Set id at first detection column.

        Find best fit between detections and previous detections.

        detections: numpy int array Cx5 [[label_id, xmin, ymin, xmax, ymax]]
        return: binded detections numpy int array Cx5 [[tracklet_id, xmin, ymin, xmax, ymax]]
        """
        detections = detections.copy()
        UNASSIGNED = -1
        for det in detections:
            det[0] = UNASSIGNED
        prev_detections = self.prev_detections

        # Write code here
        # Step 1: calc pairwise detection IOU
        ious = sorted(
            [
                (det, prev_det[0], iou_score(det[-4:], prev_det[-4:]))
                for det in detections
                for prev_det in prev_detections
            ],
            key=lambda x: x[2],
            reverse=True
        )
        used_labels = set()
        thrs = 0.7
        for det, label, iou in ious:
            if iou < thrs:
                break
            if (det[0] == UNASSIGNED) and (label not in used_labels):
                det[0] = label
                used_labels.add(label)

        # Step 2: sort IOU list

        # Step 3: fill detections[:, 0] with best match
        # One matching for each id

        # Step 4: assign new tracklet id to unmatched detections
        for det in detections:
            if det[0] == UNASSIGNED:
                det[0] = self.new_label()
        return detection_cast(detections)

    def save_detections(self, detections):
        """Save last detection frame number for each label."""
        for label in detections[:, 0]:
            self.last_detected[label] = self.frame_index

    def update_frame(self, frame):
        if not self.frame_index:
            # First frame should be processed with init_tracklet function
            detections = self.init_tracklet(frame)
        else:
            # Every Nth frame should be processed with CNN (very slow)
            # First, we extract detections
            detections = extract_detections(frame, labels=self.labels)
            # Then bind them with previous frames
            # Replacing label id to tracker id is performing in bind_tracklet function
            detections = self.bind_tracklet(detections)

        # After call CNN we save frame number for each detection
        self.save_detections(detections)
        # Save detections and frame to the history, increase frame counter
        self.detection_history.append(detections)
        self.frame_index += 1

        # Return image or raw detections
        # Image usefull to visualizing, raw detections to metric
        if self.return_images:
            return draw_detections(frame, detections)
        else:
            return detections


def main():
    dirname = os.path.dirname(__file__)
    input_clip = VideoFileClip(os.path.join(dirname, 'data', 'test.mp4'))

    tracker = Tracker()
    input_clip.fl_image(tracker.update_frame).preview()


if __name__ == '__main__':
    main()
