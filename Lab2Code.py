import numpy as np
import re
from sklearn import svm, metrics
from skimage import io, feature, filters, exposure, color
import cozmo
from cozmo.util import degrees
import time
import asyncio
import sys
from PIL import Image, ImageDraw


class ImageClassifier:

    def __init__(self):
        self.classifer = None

    def imread_convert(self, f):
        return io.imread(f).astype(np.uint8)

    def load_data_from_folder(self, dir):
        # read all images into an image collection
        ic = io.ImageCollection(dir + "*.bmp", load_func=self.imread_convert)

        # create one large array of image data
        data = io.concatenate_images(ic)

        # extract labels from image names
        labels = np.array(ic.files)
        for i, f in enumerate(labels):
            m = re.search("_", f)
            labels[i] = f[len(dir):m.start()]

        return (data, labels)

    def extract_image_features(self, data):
        l = []
        for im in data:
            im_gray = color.rgb2gray(im)

            im_gray = filters.gaussian(im_gray, sigma=0.4)

            f = feature.hog(im_gray, orientations=10, pixels_per_cell=(48, 48), cells_per_block=(4, 4),
                            feature_vector=True, block_norm='L2-Hys')
            l.append(f)

        feature_data = np.array(l)
        return (feature_data)

    def train_classifier(self, train_data, train_labels):
        self.classifer = svm.LinearSVC()
        self.classifer.fit(train_data, train_labels)

    def predict_labels(self, data):
        predicted_labels = self.classifer.predict(data)
        return predicted_labels


def main():

    img_clf = ImageClassifier()

    (train_raw, train_labels) = img_clf.load_data_from_folder('./train/')
    train_data = img_clf.extract_image_features(train_raw)
    img_clf.train_classifier(train_data, train_labels)

    def image_to_array(image):
        image_array = np.asarray(image)
        image_array.flags.writeable = True
        return image_array

    def cozmo_program(robot: cozmo.robot.Robot):
        robot.camera.image_stream_enabled = True
        robot.camera.color_image_enabled = False
        robot.camera.enable_auto_exposure()

        robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()

        while True:
            latest_image = robot.world.latest_image
            if latest_image is not None:
                new_image = latest_image.raw_image
                new_image = np.array(new_image)
                image = img_clf.extract_image_features([new_image])
                s = str(img_clf.predict_labels(image))
                robot.say_text(s)

                if "order" in s:
                    robot.drive_wheels(2,5)

                time.sleep(10)

    #time.sleep(20)

    cozmo.run_program(cozmo_program, use_viewer=True, force_viewer_on_top=True)


if __name__ == "__main__":
    main()
