import numpy as np

# ============================== 1 Classifier model ============================

def get_cls_model(input_shape):
    """
    :param input_shape: tuple (n_rows, n_cols, n_channgels)
            input shape of image for classification
    :return: nn model for classification
    """
    # your code here \/
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dropout, Dense
    from tensorflow.keras.optimizers import Adam
    model = Sequential(layers=[
        Convolution2D(32, (3, 7), activation='relu', input_shape=input_shape),
        Convolution2D(32, (3, 7), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Dropout(0.2),
        Convolution2D(32, (3, 7), activation='relu'),
        Convolution2D(32, (3, 7), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Dropout(0.2),
        Convolution2D(32, (3, 7), activation='relu'),
        Convolution2D(32, (3, 7), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dropout(0.25),
        Dense(2, activation='softmax')
    ])
    #model.summary()
    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(lr=0.0003),
        metrics=['acc']
    )
    return model
    # your code here /\

def fit_cls_model(X, y):
    """
    :param X: 4-dim ndarray with training images
    :param y: 2-dim ndarray with one-hot labels for training
    :return: trained nn model
    """
    # your code here \/
    model = get_cls_model((40, 100, 1))
    model.fit(X, y, epochs=10)
    #model.save_weights('classifier_model.h5')
    return model
    # your code here /\


# ============================ 2 Classifier -> FCN =============================
def get_detection_model(cls_model):
    """
    :param cls_model: trained cls model
    :return: fully convolutional nn model with weights initialized from cls
             model
    """
    # your code here \/
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dropout, Dense
    from tensorflow.keras.optimizers import Adam
    detection_model = Sequential(layers=[
        Convolution2D(32, (3, 7), activation='relu', input_shape=(None, None, 1)),
        Convolution2D(32, (3, 7), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Convolution2D(32, (3, 7), activation='relu'),
        Convolution2D(32, (3, 7), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Convolution2D(32, (3, 7), activation='relu'),
        Convolution2D(32, (3, 7), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Convolution2D(64, (1, 2), activation='relu'),
        Convolution2D(64, (1, 1), activation='relu'),
        Convolution2D(2, (1, 1), activation='softmax'),
    ])
    
    weights = cls_model.get_weights()

    required_shapes = []
    for layer in detection_model.layers:
        required_shapes.extend(list(map(np.shape, layer.weights)))
    required_shapes = list(map(tuple, required_shapes))

    new_weights = list(map(
        lambda w, shape: w.reshape(shape),
        weights,
        required_shapes
    ))

    detection_model.set_weights(new_weights)
    return detection_model
    # your code here /\


# ============================ 3 Simple detector ===============================
def get_detections(detection_model, dictionary_of_images):
    """
    :param detection_model: trained fully convolutional detector model
    :param dictionary_of_images: dictionary of images in format
        {filename: ndarray}
    :return: detections in format {filename: detections}. detections is a N x 5
        array, where N is number of detections. Each detection is described
        using 5 numbers: [row, col, n_rows, n_cols, confidence].
    """
    # your code here \/
    def get_one_detection(detection_model, image):
        threshold = 0.5
        prediction = detection_model.predict(image.reshape(1, *image.shape, 1))[0] > threshold
        coords = prediction[:,:,1].nonzero()
        coord_pairs = zip(coords[0], coords[1])
        
    
    common_shape = (220,370)
    for filename, image in dictionary_of_images.items():
        full_img = np.zeros(common_shape)
        h, w = image.shape
        full_img[:h, :w] = image.copy()
    return {}
    # your code here /\


# =============================== 5 IoU ========================================
def calc_iou(first_bbox, second_bbox):
    """
    :param first bbox: bbox in format (row, col, n_rows, n_cols)
    :param second_bbox: bbox in format (row, col, n_rows, n_cols)
    :return: iou measure for two given bboxes
    """
    # your code here \/
    def calc_intersection(first_bbox, second_bbox):
        l = max(first_bbox[1], second_bbox[1])
        r = min(first_bbox[1] + first_bbox[3], second_bbox[1] + second_bbox[3])
        u = max(first_bbox[0], second_bbox[0])
        d = min(first_bbox[0] + first_bbox[2], second_bbox[0] + second_bbox[2])
        return max(r - l, 0) * max(d - u, 0)
        
    def area(bbox):
        return bbox[2] * bbox[3]
    
    intersection = calc_intersection(first_bbox, second_bbox)
    return intersection / (area(first_bbox) + area(second_bbox) - intersection)
    # your code here /\


# =============================== 6 AUC ========================================
it = 1
def calc_auc(pred_bboxes, gt_bboxes):
    """
    :param pred_bboxes: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param gt_bboxes: dict of bboxes in format {filenames: bboxes}. bboxes is a
        list of tuples in format (row, col, n_rows, n_cols)
    :return: auc measure for given detections and gt
    """
    # your code here \/
    global it
    from functools import reduce
    from copy import copy
    import numpy as np
    
    confidence = lambda x: x[4]
    
    def get_tp_fp(pred, _gt):
        gt = copy(_gt)
        tp = []
        fp = []
        detections = sorted(list(map(tuple, pred)), key=confidence, reverse=True)
        for detection in detections:
            pred_bbox = detection[:4]
            if len(gt) == 0:
                fp.append(detection)
                continue
            best = reduce(
                lambda x, y: x if calc_iou(pred_bbox, x) >= calc_iou(pred_bbox, y) else y,
                gt
            )
            if calc_iou(pred_bbox, best) >= 0.5:
                tp.append(detection)
                gt.remove(best)
            else:
                fp.append(detection)
        return tp, fp
    
    tps = []
    fps = []
    for filename, pred in pred_bboxes.items():
        gt = gt_bboxes[filename]
        tp, fp = get_tp_fp(pred, gt)
        tps.extend(tp)
        fps.extend(fp)
    
    false_pos = sorted(fps, key=confidence, reverse=True)
    true_pos = sorted(tps, key=confidence, reverse=True)
    all_pred = false_pos + true_pos
    
    points = []
    n_preds = len(all_pred)
    n_true = sum(list(map(lambda x: len(x[1]), gt_bboxes.items())))
    confidences = set(list(map(confidence, all_pred)))
    for c in confidences:
        n_tp = len(list(filter(lambda x: x[4] >= c, true_pos)))
        recall = n_tp / n_true
        
        n_fp = len(list(filter(lambda x: x[4] >= c, false_pos)))
        precision = n_tp / (n_tp + n_fp)
        
        points.append((recall, precision, c))
    
    #points.append((0, 1, 1))
    #points.append((1, 0, 0))
    points = np.array(sorted(points, key=lambda x: x[2], reverse=True))
    auc = 0
    for i in range(len(points) - 1):
        auc += (points[i][1] + points[i+1][1]) / 2 * (points[i+1][0] - points[i][0])
        
    from matplotlib import pyplot as plt
    #plt.close()
    #plt.plot(points[:, 0], points[:, 1])
    #plt.ylim((0,1))
    #plt.xlim((0,1))
    #plt.savefig(str(it)+'.png')
    it += 1
        
    return 1
    # your code here /\


# =============================== 7 NMS ========================================
def nms(detections_dictionary, iou_thr=0.5):
    """
    :param detections_dictionary: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param iou_thr: IoU threshold for nearby detections
    :return: dict in same format as detections_dictionary where close detections
        are deleted
    """
    # your code here \/
    def nms_one_img(_detections):
        detections = sorted(_detections, key=lambda x: x[4], reverse=True)
        result = []
        for det in detections:
            nonmax = False
            for res in result:
                if calc_iou(det[:4], res[:4]) >= iou_thr:
                    nonmax = True
                    break
            if not nonmax:
                result.append(det)
        return result
                
    return {
        fname: nms_one_img(detections_dictionary[fname])
        for fname in detections_dictionary
    }
    # your code here /\
