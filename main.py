import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
import math
import pickle

class ImgClassifier:
    def __init__(self, srcFilePath, classNum, n_clusters):
        self.filePath = srcFilePath  # the file path of dataset
        self.classNum = classNum  # the class number of dataset
        self.k = n_clusters  # the number of cluster in kMeans
        self.trainDataset, self.trainLabel = [], []  # images, labels in train dataset
        self.testDataset, self.testLabel = [], []  # images, labels in test dataset

    def run(self):
        """Run the classifier."""
        self.initDataset()

        print('Building vocabulary with kMeans...')
        kMeansClf = KMeans(n_clusters=self.k)
        temp = self.computeSIFT(self.trainDataset, stepSize=10)
        print('Fitting classifier...')
        kMeansClf.fit(temp)
        print('Finished.')

        print('Building histogram and processing Spatial Pyramid Matching...')
        train_histogram = self.getHistogramAndSPM(2, self.trainDataset, kMeansClf)
        test_histogram = self.getHistogramAndSPM(2, self.testDataset, kMeansClf)
        print('Finished.')

        print('Train with SVM...')
        # Regularization parameter. More smaller, more regular
        clf = LinearSVC(random_state=0, C=0.001)
        clf.fit(train_histogram, self.trainLabel)
        predict = clf.predict(test_histogram)
        print("Accuracy:", np.mean(predict == self.testLabel) * 100, "%")

        f = open('predict', 'wb')
        pickle.dump({'predict': predict,
                     'label': self.testLabel},
                    f)
        f.close()


    def initDataset(self):
        """Initial the dataset."""
        print('Loading dataset:')

        for i in range(self.classNum):
            if i < 10:
                num = '0' + str(i)
            else:
                num = str(i)

            path = self.filePath + '/' + num
            print("\r{}".format(num), end='')

            for dirPath, dirNames, fileNames in os.walk(path):
                count = 0
                for index in range(len(fileNames)):
                    img = cv2.cvtColor(
                        cv2.imread(path + '/' + fileNames[index]),
                        cv2.COLOR_BGR2GRAY
                    )  # Turn images into grayscale

                    if count < 150:  # The first 150 pictures are put into train set
                        self.trainDataset.append(img)
                        self.trainLabel.append(i)
                    else:  # others are in test set
                        self.testDataset.append(img)
                        self.testLabel.append(i)
                    count += 1

        print('\rDone.')

    def computeSIFT(self, img, stepSize=15):
        """Compute the SIFT of every picture."""
        x = []
        descriptor = []
        print('Computing the SIFT...')
        for i in range(len(img)):
            strI = str(i + 1)
            if len(strI) < 4:
                strI = ' ' * (4 - len(strI)) + strI
            print('\r{}/{}'.format(strI, len(img)), end='')

            dense_sift = self.SIFT(img[i], stepSize)
            x.append(dense_sift)

        # Expand the dense_sift into vectors, every vector is 128-dimension.
        for item in x:
            for j in range(item.shape[0]):
                descriptor.append(item[j, :])

        print()
        return np.array(descriptor)

    def SIFT(self, img, stepSize=15):
        """Complete the SIFT calculation, using openCV."""
        sift = cv2.xfeatures2d.SIFT_create()
        kp = [cv2.KeyPoint(x, y, stepSize)
              for x in range(0, img.shape[0], stepSize)
              for y in range(0, img.shape[1], stepSize)]
        kp, dense_sift = sift.compute(img, kp)

        return dense_sift

    def getHistogramAndSPM(self, Level, data, kMeans):
        """Get histogram representation for training/testing data"""
        x = []
        for d in data:
            x.append(
                self.getImageFeaturesSPM(Level, d, kMeans)
            )
        return np.array(x)

    def getImageFeaturesSPM(self, Level, img, kMeans):
        H, W = img.shape
        hist = []
        for l in range(Level + 1):
            w_step = math.floor(W / (2 ** l))
            h_step = math.floor(H / (2 ** l))

            h = 0
            for i in range(0, 2 ** l):
                w = 0
                for j in range(0, 2 ** l):
                    desc = self.SIFT(img[h:h + h_step, w:w + w_step], stepSize=2)
                    predict = kMeans.predict(desc)
                    histVal = np.bincount(predict, minlength=self.k).reshape(-1)
                    hist.append(histVal * (2 ** (l - Level)))
                    w = w + w_step
                h = h + h_step

        hist = np.array(hist).reshape(-1)
        return (hist - np.mean(hist)) / np.std(hist)

if __name__ == '__main__':
    example = ImgClassifier(srcFilePath='15-Scene', classNum=15, n_clusters=60)
    example.run()
