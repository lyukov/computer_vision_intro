
def iou_score(bbox1, bbox2):
    """Jaccard index or Intersection over Union.

    https://en.wikipedia.org/wiki/Jaccard_index

    bbox: [xmin, ymin, xmax, ymax]
    """

    assert len(bbox1) == 4
    assert len(bbox2) == 4

    # Write code here

    def calc_intersection(bbox1, bbox2):
        l = max(bbox1[0], bbox2[0])
        r = min(bbox1[2], bbox2[2])
        u = max(bbox1[1], bbox2[1])
        d = min(bbox1[3], bbox2[3])
        return max(r - l, 0) * max(d - u, 0)
        
    def area(bbox):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
    intersection = calc_intersection(bbox1, bbox2)
    return intersection / (area(bbox1) + area(bbox2) - intersection)


def motp(obj, hyp, threshold=0.5):
    """Calculate MOTP

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """
    obj = obj.copy()
    hyp = hyp.copy()

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Write code here

        # Step 1: Convert frame detections to dict with IDs as keys
        dict_obj = dict(list(map(
            lambda x: (x[0], x[1:]),
            frame_obj
        )))
        dict_hyp = dict(list(map(
            lambda x: (x[0], x[1:]),
            frame_hyp
        )))

        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        for obj_id, hyp_id in matches.items():
            if (obj_id not in dict_obj) or (hyp_id not in dict_hyp):
                continue
            iou = iou_score(dict_obj[obj_id], dict_hyp[hyp_id])
            if iou > threshold:
                del dict_obj[obj_id]
                del dict_hyp[hyp_id]
                dist_sum += iou
                match_count += 1

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        ious = list(filter(
            lambda x: x[2] > threshold,
            sorted(
                [
                    (obj_id, hyp_id, iou_score(obj_bbox, hyp_bbox))
                    for obj_id, obj_bbox in dict_obj.items()
                    for hyp_id, hyp_bbox in dict_hyp.items()
                ],
                key=lambda x: x[2],
                reverse=True
            )
        ))

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        for obj_id, hyp_id, iou in ious:
            matches[obj_id] = hyp_id
            dist_sum += iou
            match_count += 1

        # Step 5: Update matches with current matched IDs
        pass

    # Step 6: Calculate MOTP
    MOTP = dist_sum / match_count

    return MOTP


def motp_mota(obj, hyp, threshold=0.5):
    """Calculate MOTP/MOTA

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0
    missed_count = 0
    false_positive = 0
    mismatch_error = 0
    present_count = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        present_count += len(frame_obj)
        # Step 1: Convert frame detections to dict with IDs as keys
        dict_obj = dict(list(map(
            lambda x: (x[0], x[1:]),
            frame_obj
        )))
        dict_hyp = dict(list(map(
            lambda x: (x[0], x[1:]),
            frame_hyp
        )))
        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        for obj_id, hyp_id in matches.items():
            if (obj_id not in dict_obj) or (hyp_id not in dict_hyp):
                continue
            iou = iou_score(dict_obj[obj_id], dict_hyp[hyp_id])
            if iou > threshold:
                del dict_obj[obj_id]
                del dict_hyp[hyp_id]
                dist_sum += iou
                match_count += 1
        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        ious = list(filter(
            lambda x: x[2] > threshold,
            sorted(
                [
                    (obj_id, hyp_id, iou_score(obj_bbox, hyp_bbox))
                    for obj_id, obj_bbox in dict_obj.items()
                    for hyp_id, hyp_bbox in dict_hyp.items()
                ],
                key=lambda x: x[2],
                reverse=True
            )
        ))
        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        for obj_id, hyp_id, iou in ious:
            if obj_id in matches:
                if matches[obj_id] != hyp_id:
                    mismatch_error += 1
            matches[obj_id] = hyp_id
            del dict_obj[obj_id]
            del dict_hyp[hyp_id]
            dist_sum += iou
            match_count += 1
        # Step 5: If matched IDs contradict previous matched IDs - increase mismatch error

        # Step 6: Update matches with current matched IDs

        # Step 7: Errors
        # All remaining hypotheses are considered false positives
        # All remaining objects are considered misses
        missed_count += len(dict_obj)
        false_positive += len(dict_hyp)

    # Step 8: Calculate MOTP and MOTA
    MOTP = dist_sum / match_count
    MOTA = 1 - (missed_count + false_positive + mismatch_error) / present_count

    return MOTP, MOTA
