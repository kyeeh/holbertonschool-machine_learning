#!/usr/bin/env python3
"""
Object Detection Module
"""
import numpy as np
import tensorflow.keras as K


class Yolo:
    """
    Yolo class uses the Yolo v3 algorithm to perform object detection
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Yolo instance constructor

        model_path is the path to where a Darknet Keras model is stored
        classes_path is the path to where the list of class names used for the
        Darknet model, listed in order of index, can be found
        class_t is a float representing the box score threshold for the initial
        filtering step
        nms_t is a float representing the IOU threshold for non-max suppression
        anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
        containing all of the anchor boxes:
            outputs is the number of outputs (predictions) made by the Darknet
            model anchor_boxes is the number of anchor boxes used for each
            prediction
            2 => [anchor_box_width, anchor_box_height]

        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = f.read().splitlines()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def _sigmoid(self, x):
        """
        Sigmoid function
        """
        return (1 / (1 + np.exp(-x)))

    def process_outputs(self, outputs, image_size):
        """
        outputs is a list of numpy.ndarrays containing the predictions from
        the Darknet model for a single image:
            Each output will have the shape (grid_height, grid_width,
            anchor_boxes, 4 + 1 + classes)
                grid_height & grid_width => the height and width of the grid
                used for the output
                anchor_boxes => the number of anchor boxes used
                4 => (t_x, t_y, t_w, t_h)
                1 => box_confidence
                classes => class probabilities for all classes
        image_size is a numpy.ndarray containing the image’s original size
        [image_height, image_width]


        Returns a tuple of (boxes, box_confidences, box_class_probs):
            boxes: a list of numpy.ndarrays of shape (grid_height, grid_width,
            anchor_boxes, 4) containing the processed boundary boxes for each
            output, respectively:
                4 => (x1, y1, x2, y2)
                (x1, y1, x2, y2) should represent the boundary box relative
                to original image
            box_confidences: a list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, 1) containing the box confidences for
            each output, respectively
            box_class_probs: a list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, classes) containing the box’s class
            probabilities for each output, respectively
        """
        boxes = []

        for i, output in enumerate(outputs):
            img_h, img_w = image_size
            grid_h, grid_w, num_boxes, _ = output.shape

            box_xy = self._sigmoid(output[:, :, :, :2])

            box_mag = np.exp(output[:, :, :, 2:4])
            anchors = self.anchors.reshape(1, 1, self.anchors.shape[0],
                                           num_boxes, 2)
            box_mag *= anchors[:, :, i, :, :]

            corner_x = np.tile(np.arange(0, grid_w),
                               grid_w).reshape(grid_w, grid_w)
            corner_y = np.tile(np.arange(0, grid_h),
                               grid_w).reshape(grid_w, grid_h).T

            corner_x = corner_x.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=2)
            corner_y = corner_y.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=2)

            corner_xy = np.concatenate((corner_x, corner_y), axis=3)

            box_xy += corner_xy
            box_xy /= (grid_w, grid_h)
            box_mag /= (self.model.input.shape[1].value,
                        self.model.input.shape[2].value)
            box_xy -= (box_mag / 2)

            box = np.concatenate((box_xy, box_xy + box_mag), axis=-1)

            box[..., 0] *= img_w
            box[..., 1] *= img_h
            box[..., 2] *= img_w
            box[..., 3] *= img_h

            boxes.append(box)

        box_prob = [self._sigmoid(out[..., 5:]) for out in outputs]
        box_confidence = [self._sigmoid(out[..., 4:5]) for out in outputs]

        return (boxes, box_confidence, box_prob)
