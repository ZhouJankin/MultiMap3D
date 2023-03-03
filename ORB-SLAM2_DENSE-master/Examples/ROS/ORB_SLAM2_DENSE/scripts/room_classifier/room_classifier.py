import pickle
import os.path
from sklearn import model_selection as cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer

from time import time
from sklearn.svm import SVC

class RoomClassifier:
  def __init__(self):
    # read our labels from a pickle file
    self.labels_fname = "labels_shuffled.pkl"
    self.features_fname = "features_for_each_label.pkl"

    file = open(self.features_fname,'rb')
    features_for_each_label = pickle.load(file)
    file.close()

    file = open(self.labels_fname,'rb')
    labels_shuffled = pickle.load(file)
    file.close()

    # Now create training and testing sets
    features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features_for_each_label, labels_shuffled, test_size=0.1, random_state=1983)

    # Now we will turn the texts into numerical vectors so that we can use that for machine learning
    #vectorizer = TfidfVectorizer(max_df=1.0, stop_words='english')
    self.vectorizer = TfidfVectorizer()

    # If we want to give all features and all labels to training (leaving no unique test cases), then uncomment below
    #features_train = features_for_each_label
    #labels_train = labels_shuffled

    features_train_vectorized = self.vectorizer.fit_transform(features_train)
    self.features_test_vectorized  = self.vectorizer.transform(features_test)

    #print "tfidf.get_stop_words(): ",tfidf.get_stop_words()
    #print "vector: ",vector
    print(features_train_vectorized.shape)
    print(self.features_test_vectorized.shape)
    print(self.vectorizer.get_feature_names_out())


    # Finally learn and test how good model have we got
    self.clf = SVC(kernel="rbf", C=10000.0)

    #features_train = features_train[:len(features_train)/100]
    #labels_train = labels_train[:len(labels_train)/100]

    t0 = time()
    self.clf.fit(features_train_vectorized, labels_train)
    print("training time:", round(time()-t0, 3), "s")

    t0 = time()
    self.pred = self.clf.predict(self.features_test_vectorized)
    print("prediction time:", round(time()-t0, 3), "s")

    from sklearn.metrics import accuracy_score

    to = time()
    acc = accuracy_score(self.pred, labels_test)
    print("accuracy calculation time:", round(time()-t0, 3), "s")

    print("Accuracy = ",acc)

    print("Predicted Class for Elem 10:",self.pred[10]," Class for Elem 8:",self.pred[8]," Class for elem 5:", self.pred[5])

    print("Real Class for Elem 10:",labels_test[10]," Real Class for Elem 8:",labels_test[8]," Real Class for elem 5:", labels_test[5])
    
    print(self.clf.classes_)
    #########################################################

  def predict(self, items_as_string_separated_by_space):
    input_vectorized  = self.vectorizer.transform([items_as_string_separated_by_space])
    #print(input_vectorized)
    result = self.clf.predict(input_vectorized)
    
    print("Prediction of: " + items_as_string_separated_by_space + " : " + result[0])

rc = RoomClassifier()
rc.predict("SinkBasin CounterTop SoapBar ToiletPaperHanger")
rc.predict("SinkBasin Chair Egg Toaster Microwave CounterTop DiningTable StoveKnob Lettuce SaltShaker")
rc.predict("SinkBasin Chair Egg Toaster Microwave CounterTop DiningTable StoveKnob Lettuce")
rc.predict("Egg")
#rc.predict("SinkBasin CounterTop SoapBar ToiletPaperHanger ToiletPaper SprayBottle Floor GarbageCan Candle Plunger ScrubBrush Toilet Sink HandTowelHolder Faucet Mirror Cloth Towel Drawer SoapBottle ShowerHead HandTowel LightSwitch ShowerDoor TowelHolder ShowerGlass")
#rc.predict("Candle Plunger ScrubBrush Toilet Sink HandTowelHolder SoapBottle")
rc.predict("Candle Plunger ScrubBrush Toilet")
rc.predict("TV Sofa")
rc.predict("ScrubBrush ToiletCandle Plunger")


