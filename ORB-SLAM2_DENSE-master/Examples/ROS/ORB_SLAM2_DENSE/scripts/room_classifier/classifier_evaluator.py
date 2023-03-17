##
# This module will create multiple data sets using room_labels_and_features_generator.py and then
# train SVCs on those data sets using room_classifier.py and collect the accuracies.
##

from room_classifier import RoomClassifier
from room_labels_and_features_generator import RoomClassifierTrainingDataGenerator
from ModelType import ModelType

accuracies = []
for i in range(20):
    gen = RoomClassifierTrainingDataGenerator(ModelType.FEATURES_12, 1100)
    rc = RoomClassifier(True, ModelType.FEATURES_12)
    accuracies.append(rc.getAccuracy())

print(accuracies)
