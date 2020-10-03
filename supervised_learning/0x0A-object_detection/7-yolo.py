#!/usr/bin/env python3
"""
Object Detection Module
"""
import cv2
import glob
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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        boxes: a list of numpy.ndarrays of shape (grid_height, grid_width,
        anchor_boxes, 4) containing the processed boundary boxes for each
        output, respectively
        box_confidences: a list of numpy.ndarrays of shape (grid_height,
        grid_width, anchor_boxes, 1) containing the processed box
        confidences for each output, respectively
        box_class_probs: a list of numpy.ndarrays of shape (grid_height,
        grid_width, anchor_boxes, classes) containing the processed box class
        probabilities for each output, respectively

        Returns a tuple of (filtered_boxes, box_classes, box_scores):
            filtered_boxes: a numpy.ndarray of shape (?, 4) containing all of
            the filtered bounding boxes:
            box_classes: a numpy.ndarray of shape (?,) containing the class
            number that each box in filtered_boxes predicts, respectively
            box_scores: a numpy.ndarray of shape (?) containing the box scores
            for each box in filtered_boxes, respectively
        """
        scores = []
        for i in range(len(boxes)):
            scores.append(box_class_probs[i] * box_confidences[i])

        class_idx = [np.argmax(elem, -1) for elem in scores]
        class_idx = [elem.reshape(-1) for elem in class_idx]
        class_idx = np.concatenate(class_idx)

        class_score = [np.max(elem, axis=-1) for elem in scores]
        class_score = [elem.reshape(-1) for elem in class_score]
        class_score = np.concatenate(class_score)

        filter_box = [elem.reshape(-1, 4) for elem in boxes]
        filter_box = np.concatenate(filter_box)

        filter = np.where(class_score > self.class_t)

        box_classes = class_idx[filter]
        box_scores = class_score[filter]
        filter_boxes = filter_box[filter]

        return (filter_boxes, box_classes, box_scores)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        filtered_boxes: a numpy.ndarray of shape (?, 4) containing all of the
        filtered bounding boxes:
        box_classes: a numpy.ndarray of shape (?,) containing the class number
        for the class that filtered_boxes predicts, respectively
        box_scores: a numpy.ndarray of shape (?) containing the box scores for
        each box in filtered_boxes, respectively

        Returns a tuple of (box_predictions, predicted_box_classes,
        predicted_box_scores):
            box_predictions: a numpy.ndarray of shape (?, 4) containing all of
            the predicted bounding boxes ordered by class and box score
            predicted_box_classes: a numpy.ndarray of shape (?,) containing
            the class number for box_predictions ordered by class and box
            score, respectively
            predicted_box_scores: a numpy.ndarray of shape (?) containing the
            box scores for box_predictions ordered by class and box score,
            respectively
        """
        boxes, classes, scores = [], [], []

        for klasses in set(box_classes):
            index = np.where(box_classes == klasses)

            klass = box_classes[index]
            bscores = box_scores[index]
            bxs = filtered_boxes[index]

            x1, x2 = bxs[:, 0], bxs[:, 2]
            y1, y2 = bxs[:, 1], bxs[:, 3]

            a = (x2 - x1 + 1) * (y2 - y1 + 1)
            index = bscores.argsort()[::-1]
            keep = []

            while index.size > 0:
                i = index[0]
                j = index[1:]
                keep.append(i)

                yy1 = np.maximum(y1[i], y1[j])
                xx1 = np.maximum(x1[i], x1[j])
                yy2 = np.minimum(y2[i], y2[j])
                xx2 = np.minimum(x2[i], x2[j])

                w = np.maximum(0.0, xx2 - xx1 + 1)
                h = np.maximum(0.0, yy2 - yy1 + 1)

                box_area = (w * h)
                overlap = box_area / (a[i] + a[j] - box_area)
                index = index[(np.where(overlap <= self.nms_t)[0]) + 1]

            keep = np.array(keep)

            boxes.append(bxs[keep])
            classes.append(klass[keep])
            scores.append(bscores[keep])

        filtered_boxes = np.concatenate(boxes)
        box_classes = np.concatenate(classes)
        box_scores = np.concatenate(scores)

        return (filtered_boxes, box_classes, box_scores)

    @staticmethod
    def load_images(folder_path):
        """
        Add the static method def load_images(folder_path):

            folder_path: a string representing the path to the folder holding
            all the images to load

            Returns a tuple of (images, image_paths):
                images: a list of images as numpy.ndarrays
                image_paths: a list of paths to the individual img in images
        """
        image_paths = glob.glob(folder_path + "/*")
        images = [cv2.imread(img) for img in image_paths]
        return (images, image_paths)

    def preprocess_images(self, images):
        """
        images: a list of images as numpy.ndarrays
        Resize the images with inter-cubic interpolation
        Rescale all images to have pixel values in the range [0, 1]

        Returns a tuple of (pimages, image_shapes):
            pimages: a numpy.ndarray of shape (ni, input_h, input_w, 3)
            containing all of the preprocessed images
                ni: the number of images that were preprocessed
                input_h: the input height for the Darknet model
                input_w: the input width for the Darknet model
                Note: this can vary by model
                3: number of color channels
            image_shapes: a numpy.ndarray of shape (ni, 2) containing the
            original height and width of the images
                2 => (image_height, image_width)
        """
        rsized_images = []
        input_w = self.model.input.shape[1].value
        input_h = self.model.input.shape[2].value

        ishapes = [img.shape[:2] for img in images]
        img_shapes = np.stack(ishapes, axis=0)

        for image in images:
            rsized_img = cv2.resize(image, (input_w, input_h),
                                    interpolation=cv2.INTER_CUBIC)
            rscaled_img = rsized_img / 255
            rsized_images.append(rscaled_img)

        rsized_images = np.stack(rsized_images, axis=0)
        return (rsized_images, img_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        image: a numpy.ndarray containing an unprocessed image
        boxes: a numpy.ndarray containing the boundary boxes for the image
        box_classes: a numpy.ndarray containing the class indices for each box
        box_scores: a numpy.ndarray containing the box scores for each box
        file_name: the file path where the original image is stored

        Displays the image with all boundary boxes, class names, and box
        scores (see example below)
            Boxes should be drawn as with a blue line of thickness 2
            Class names and box scores should be drawn above each box in red
                Box scores should be rounded to 2 decimal places
                Text should be written 5 pixels above the top left corner of
                the box
                Text should be written in FONT_HERSHEY_SIMPLEX
                Font scale should be 0.5
                Line thickness should be 1
                You should use LINE_AA as the line type
            The window name should be the same as file_name
            If the s key is pressed:
                The image should be saved in the directory detections, located
                in the current directory
                If detections does not exist, create it
                The saved image should have the file name file_name
                The image window should be closed
            If any key besides s is pressed, the image window should be closed
            without saving
        """
        for i, box in enumerate(boxes):
            cl = (255, 0, 0)
            cp = (0, 0, 255)
            sp = (int(box[0]), int(box[3]))
            ep = (int(box[2]), int(box[1]))
            text_box = (int(box[0]), int(box[1])-5)
            cv2.rectangle(image, sp, ep, cl, thickness=2)
            cv2.putText(image, self.class_names[box_classes[i]] + " " +
                        "{:.2f}".format(box_scores[i]), text_box,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, cp, thickness=1,
                        lineType=cv2.LINE_AA, bottomLeftOrigin=False)
        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)
        if key == ord('s'):
            if not os.path.exists('detections'):
                os.makedirs('detections')
            os.chdir('detections')
            cv2.imwrite(file_name, image)
            os.chdir('../')
        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        Predict function
        """
        prd = []
        images, image_paths = self.load_images(folder_path)
        return (prd, image_paths)
