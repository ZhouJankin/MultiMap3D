import pickle
import os.path
from sklearn import model_selection as cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer

from time import time
from sklearn.svm import SVC
from room_classifier.ModelType import ModelType

class RoomClassifier:
  ###
  # Initialises the SVC- either by training from the training data or by loading it from 
  # a pickle file- depending on the TRAIN_FROM_SCRATCH parameter.
  ###
  def __init__(self, TRAIN_FROM_SCRATCH = False, detector_type = ModelType.FEATURES_18):
  
    self.load_training_data_and_vectorize(detector_type)
  
    # We can either load pre-trained settings or train from scratch
    if TRAIN_FROM_SCRATCH:
        # We will use an Support Vector Classifier
        self.clf = SVC(kernel="rbf", C=10000.0)

        #features_train = features_train[:len(features_train)/100]
        #labels_train = labels_train[:len(labels_train)/100]

        # Now let's train our SVC
        t0 = time()
        self.clf.fit(self.features_train_vectorized, self.labels_train)
        print("training time:", round(time()-t0, 3), "s")
        
        # Now let's save our trained parameters
        pickle.dump(self.clf, open("trained_svc_" + detector_type.name + ".pkl", "wb"))

    else: # so we want to load pre-trained parameters
        file = open("trained_svc_" + detector_type.name + ".pkl",'rb')
        self.clf = pickle.load(file)
        file.close()
            
            
    t0 = time()
    self.pred = self.clf.predict(self.features_test_vectorized)
    print("prediction time:", round(time()-t0, 3), "s")

    from sklearn.metrics import accuracy_score

    # Finally learn and test how good model have we got
    t0 = time()
    acc = accuracy_score(self.pred, self.labels_test)
    print("accuracy calculation time:", round(time()-t0, 3), "s")

    print("Accuracy = ",acc)

    print("Predicted Class for Elem 3:",self.pred[3]," Class for Elem 8:",self.pred[8]," Class for elem 5:", self.pred[5])

    print("Real Class for Elem 3:",self.labels_test[3]," Real Class for Elem 8:",self.labels_test[8]," Real Class for elem 5:", self.labels_test[5])
    print(self.clf.classes_)
    #########################################################

  ###
  # Loads training data from pickle files and vecorizes the data for use in the SVC
  ###  
  def load_training_data_and_vectorize(self, detector_type):
    # read our labels from a pickle file
    self.labels_fname = "room_classifier/labels_shuffled_" + detector_type.name + ".pkl"
    self.features_fname = "room_classifier/features_for_each_label_" + detector_type.name + ".pkl"

    file = open(self.features_fname,'rb')
    features_for_each_label = pickle.load(file)
    file.close()

    file = open(self.labels_fname,'rb')
    labels_shuffled = pickle.load(file)
    file.close()

    # Now create training and testing sets
    features_train, features_test, self.labels_train, self.labels_test = cross_validation.train_test_split(features_for_each_label, labels_shuffled, test_size=0.1, random_state=1983)

    # Now we will turn the texts into numerical vectors so that we can use that for machine learning
    #vectorizer = TfidfVectorizer(max_df=1.0, stop_words='english')
    self.vectorizer = TfidfVectorizer()

    # If we want to give all features and all labels to training (leaving no unique test cases), then uncomment below
    #features_train = features_for_each_label
    #labels_train = labels_shuffled

    self.features_train_vectorized = self.vectorizer.fit_transform(features_train)
    self.features_test_vectorized  = self.vectorizer.transform(features_test)

    #print "tfidf.get_stop_words(): ",tfidf.get_stop_words()
    #print "vector: ",vector
    print(self.features_train_vectorized.shape)
    print(self.features_test_vectorized.shape)
    print(self.vectorizer.get_feature_names_out())

  ###
  # Uses the cassifier to predict a room based on the input elements found in the room
  ###
  def predict(self, items_as_string_separated_by_space):
    input_vectorized  = self.vectorizer.transform([items_as_string_separated_by_space])
    #print(input_vectorized)
    result = self.clf.predict(input_vectorized)
    print("Prediction of: " + items_as_string_separated_by_space + " : " + result[0])
    return result[0]


rc = RoomClassifier(True, ModelType.AI2_THOR_12)
rc.predict("SinkBasin CounterTop SoapBar ToiletPaperHanger")
rc.predict("SinkBasin Chair Egg Toaster Microwave CounterTop DiningTable StoveKnob Lettuce SaltShaker")
rc.predict("SinkBasin Chair Egg Toaster Microwave CounterTop DiningTable StoveKnob Lettuce")
rc.predict("Egg")
#rc.predict("SinkBasin CounterTop SoapBar ToiletPaperHanger ToiletPaper SprayBottle Floor GarbageCan Candle Plunger ScrubBrush Toilet Sink HandTowelHolder Faucet Mirror Cloth Towel Drawer SoapBottle ShowerHead HandTowel LightSwitch ShowerDoor TowelHolder ShowerGlass")
#rc.predict("Candle Plunger ScrubBrush Toilet Sink HandTowelHolder SoapBottle")
rc.predict("Candle Plunger ScrubBrush Toilet")
rc.predict("TV Sofa")
rc.predict("ScrubBrush ToiletCandle Plunger")

rc.predict("sink garbagebin door cabinet counter refrigerator window")
rc.predict("window table bed desk")
rc.predict("sofa door table window bookshelf curtain desk chair picture")
rc.predict("door picture window curtain")
rc.predict("garbagebin counter refrigerator cabinet")

