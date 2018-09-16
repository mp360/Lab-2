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
        feature_data = []
        for image in data:
            converted_image = color.rgb2gray(image)
            exp = exposure.adjust_gamma(converted_image, gamma=1.5, gain=1.5)
            gauss = filters.gaussian(exp)
            features = feature.hog(gauss, orientations=10, pixels_per_cell=(11, 11), cells_per_block=(3, 3))
            feature_data.append(features)


        return (feature_data)

    def train_classifier(self, train_data, train_labels):
        self.classifer = svm.LinearSVC()
        self.classifer.fit(train_data, train_labels)

    def predict_labels(self, data):
        predicted_labels = self.classifer.predict(data)
        return predicted_labels

class Actions:

    classifier = ImageClassifier()

    def say_image(self, image):
        a = ""
        a += classifier.predict_labels(str(image))
        return a




def main():

    img_clf = ImageClassifier()
    actions = Actions()

    (train_raw, train_labels) = img_clf.load_data_from_folder('./Fall_2018_Class_Images/')
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
            new_image = np.array(latest_image.raw_image)
            #new_image = color.rgb2gray(new_image)
            new_image = img_clf.extract_image_features(new_image)
            s = img_clf.predict_labels(new_image)
            print(s)

    #time.sleep(20)

    cozmo.run_program(cozmo_program, use_viewer=True, force_viewer_on_top=True)


if __name__ == "__main__":
    main()