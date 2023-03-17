import random
import pickle

from ModelType import ModelType

class RoomClassifierTrainingDataGenerator:
  ###
  # Initialises the training data generator and attempts to generate the required data based on the gen_flavour parameter.
  # gen_flavour: what datasets we want to generate (choices in ModelType.py)
  # datasets_to_gen: how many datasets we want. This is only applicable to generated datasets (FEATURES_12 and FEATURES_18) The translated ones (AI2-THOR ones) are always 120
  # and hybridized ones are whatever was generated for FEATURES_12 or FEATURES_18 + the 120 of AI2-THOR.
  ###
  def __init__(self, gen_flavour, datasets_to_gen = 1000):
    print("Generating for: " + gen_flavour.name)
    if gen_flavour == ModelType.FEATURES_12:
        # So we have these features - that's all we know how to detect.
        # Let's construct a few labels (room types) and assign some of these items to them
        room_types = ['bedroom', 'bathroom', 'study', 'living_room']

        detectable_items = ['bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
                       'night_stand', 'bookshelf', 'bathtub']

        possible_items_in_each_room = [
        (room_types[0], ['bed', 'table', 'chair', 'dresser', 'night_stand', 'desk', 'bookshelf']),
        (room_types[1], ['toilet', 'bathtub']),
        (room_types[2], ['desk', 'chair', 'bookshelf']),
        (room_types[3], ['table', 'sofa', 'chair', 'bookshelf', 'dresser', 'desk'])
        ]
        
        self.TRAINING_DATASETS_TO_GENERATE = datasets_to_gen
        self.generate_training_data(gen_flavour, room_types, possible_items_in_each_room)
    elif gen_flavour == ModelType.FEATURES_18:
        # Alternative model. We're missing dresser and nightstand and have a few extra features: 
        # cabinet, door, window, picture, counter, curtain, refrigerator, showercurtrain, sink, garbagebin
        room_types = ['bedroom', 'bathroom', 'study', 'living_room', 'kitchen']

        detectable_items = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                        'bookshelf', 'picture', 'counter', 'desk', 'curtain',
                        'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
                        'garbagebin']

        possible_items_in_each_room = [
        (room_types[0], ['bed', 'table', 'chair', 'desk', 'bookshelf', 'cabinet', 'door', 'window', 'picture', 'curtain']),
        (room_types[1], ['toilet', 'bathtub', 'garbagebin', 'door', 'window', 'counter', 'showercurtrain', 'sink']),
        (room_types[2], ['desk', 'chair', 'bookshelf', 'door', 'window', 'cabinet', 'picture', 'curtain', 'garbagebin']),
        (room_types[3], ['table', 'sofa', 'chair', 'bookshelf', 'desk', 'door', 'window', 'picture', 'curtain']),
        (room_types[4], ['cabinet', 'chair', 'door', 'window', 'counter', 'refrigerator', 'sink', 'garbagebin'])
        ]
        
        #self.TRAINING_DATASETS_TO_GENERATE = 360
        self.TRAINING_DATASETS_TO_GENERATE = datasets_to_gen
        self.generate_training_data(gen_flavour, room_types, possible_items_in_each_room)
    elif gen_flavour == ModelType.AI2_THOR_12:
        detectable_items = ['bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
                       'night_stand', 'bookshelf', 'bathtub'] # these items we can detect with our framework
                       
        available_items = ['aluminumfoil', 'alarmclock', 'apple', 'armchair', 'baseballbat', 'basketball', 'bathtub',
                     'bathtubbasin', 'bed', 'blinds', 'book', 'boots', 'bottle', 'bowl', 'box',
                     'bread', 'butterknife', 'cabinet', 'candle', 'cd', 'cellphone', 'chair', 'cloth',
                     'coffeemachine', 'coffeetable', 'countertop', 'creditcard', 'cup', 'curtains',
                     'desk', 'desklamp', 'desktop', 'diningtable', 'dishsponge', 'dogbed', 'drawer',
                     'dresser', 'dumbbell', 'egg', 'faucet', 'floor', 'floorlamp', 'footstool',
                     'fork', 'fridge', 'garbagebag', 'garbagecan', 'handtowel', 'handtowelholder',
                     'houseplant', 'kettle', 'keychain', 'knife', 'ladle', 'laptop', 'laundryhamper',
                     'lettuce', 'lightswitch', 'microwave', 'mirror', 'mug', 'newspaper', 'ottoman',
                     'painting', 'pan', 'papertowelroll', 'pen', 'pencil', 'peppershaker', 'pillow',
                     'plate', 'plunger', 'poster', 'pot', 'potato', 'remotecontrol', 'roomdecor',
                     'safe', 'saltshaker', 'scrubbrush', 'shelf', 'shelvingunit', 'showercurtain',
                     'showerdoor', 'showerglass', 'showerhead', 'sidetable', 'sink', 'sinkbasin',
                     'soapbar', 'soapbottle', 'sofa', 'spatula', 'spoon', 'spraybottle', 'statue',
                     'stool', 'stoveburner', 'stoveknob', 'tabletopdecor', 'teddybear',
                     'television', 'tennisracket', 'tissuebox', 'toaster', 'toilet', 'toiletpaper',
                     'toiletpaperhanger', 'tomato', 'towel', 'towelholder', 'tvstand',
                     'vacuumcleaner', 'vase', 'watch', 'wateringcan', 'window', 'winebottle'] # these items are available in AI2-THOR framework
                     
        items_to_convert = ['armchair', 'bathtubbasin', 'coffeetable', 
                     'desktop', 'diningtable', 'sidetable', 'shelf', 'shelvingunit'] # these items we will need to convert from AI2-THOR vocabulary to ours
        converting_to = ['chair', 'bathtub', 'table', 'desk', 'table', 'table', 'bookshelf', 'bookshelf'] # We will convert AI2-THOR vocabulary to this one
                     
        items_transferred_directly = ['bed', 'bathtub', 'chair', 'desk', 'dresser', 'sofa', 'toilet'] # These items are the same in our vocabulary and AI2-THOR. missing: night_stand
                     
        items_to_remove = ['aluminumfoil', 'alarmclock', 'apple', 'baseballbat', 'basketball', 'blinds', 'book', 'boots', 'bottle', 'bowl', 'box',
                     'bread', 'butterknife', 'cabinet', 'candle', 'cd', 'cellphone', 'cloth',
                     'coffeemachine', 'creditcard', 'cup', 'desklamp', 'dishsponge', 'dogbed', 'drawer', 'dumbbell', 'egg', 'faucet', 'floor', 'floorlamp', 'footstool',
                     'fork', 'fridge', 'garbagebag', 'garbagecan', 'handtowel', 'handtowelholder',
                     'houseplant', 'kettle', 'keychain', 'knife', 'ladle', 'laptop', 'laundryhamper',
                     'lettuce', 'lightswitch', 'microwave', 'mirror', 'mug', 'newspaper', 'ottoman',
                     'painting', 'pan', 'papertowelroll', 'pen', 'pencil', 'peppershaker', 'pillow',
                     'plate', 'plunger', 'poster', 'pot', 'potato', 'remotecontrol', 'roomdecor',
                     'safe', 'saltshaker', 'scrubbrush', 'shelf', 'showercurtain',
                     'showerdoor', 'showerglass', 'showerhead', 'soapbar', 'soapbottle', 'spatula', 'spoon', 'spraybottle', 'statue',
                     'stool', 'stoveburner', 'stoveknob', 'tabletopdecor', 'teddybear',
                     'television', 'tennisracket', 'tissuebox', 'toaster', 'toiletpaper',
                     'toiletpaperhanger', 'tomato', 'towel', 'towelholder', 'tvstand',
                     'vacuumcleaner', 'vase', 'watch', 'wateringcan', 'window', 'winebottle', 'countertop', 'curtains', 'sink', 'sinkbasin'] # These items we will need to remove from AI2-THOR provided set because our model can't detect them.
                     
        room_types = ['bedroom', 'bathroom', 'living room'] # With the detectable models that we have, we'll be able to detect these rooms. There is nothing to detect for kitchen, so dropping that from AI2-THOR room labels.

        self.convert_ai2_thor_training_data(gen_flavour, room_types, items_to_remove, items_to_convert, converting_to)
    elif gen_flavour == ModelType.AI2_THOR_18:
        detectable_items = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                        'bookshelf', 'picture', 'counter', 'desk', 'curtain',
                        'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
                        'garbagebin'] # these items we can detect with our framework
                                            
        available_items = ['aluminumfoil', 'alarmclock', 'apple', 'armchair', 'baseballbat', 'basketball', 'bathtub',
                     'bathtubbasin', 'bed', 'blinds', 'book', 'boots', 'bottle', 'bowl', 'box',
                     'bread', 'butterknife', 'cabinet', 'candle', 'cd', 'cellphone', 'chair', 'cloth',
                     'coffeemachine', 'coffeetable', 'countertop', 'creditcard', 'cup', 'curtains',
                     'desk', 'desklamp', 'desktop', 'diningtable', 'dishsponge', 'dogbed', 'drawer',
                     'dresser', 'dumbbell', 'egg', 'faucet', 'floor', 'floorlamp', 'footstool',
                     'fork', 'fridge', 'garbagebag', 'garbagecan', 'handtowel', 'handtowelholder',
                     'houseplant', 'kettle', 'keychain', 'knife', 'ladle', 'laptop', 'laundryhamper',
                     'lettuce', 'lightswitch', 'microwave', 'mirror', 'mug', 'newspaper', 'ottoman',
                     'painting', 'pan', 'papertowelroll', 'pen', 'pencil', 'peppershaker', 'pillow',
                     'plate', 'plunger', 'poster', 'pot', 'potato', 'remotecontrol', 'roomdecor',
                     'safe', 'saltshaker', 'scrubbrush', 'shelf', 'shelvingunit', 'showercurtain',
                     'showerdoor', 'showerglass', 'showerhead', 'sidetable', 'sink', 'sinkbasin',
                     'soapbar', 'soapbottle', 'sofa', 'spatula', 'spoon', 'spraybottle', 'statue',
                     'stool', 'stoveburner', 'stoveknob', 'tabletopdecor', 'teddybear',
                     'television', 'tennisracket', 'tissuebox', 'toaster', 'toilet', 'toiletpaper',
                     'toiletpaperhanger', 'tomato', 'towel', 'towelholder', 'tvstand',
                     'vacuumcleaner', 'vase', 'watch', 'wateringcan', 'window', 'winebottle'] # these items are available in AI2-THOR framework
                     
        items_to_convert = ['armchair', 'bathtubbasin', 'coffeetable', 'countertop', 'curtains',
                     'desktop', 'diningtable', 'fridge', 'garbagecan', 
                     'painting', 'poster', 'shelf', 'shelvingunit', 'showerdoor', 'sidetable', 'sinkbasin', 'showercurtain'] # these items we will need to convert from AI2-THOR vocabulary to ours
        converting_to = ['chair', 'bathtub', 'table', 'counter', 'curtain', 
                     'desk', 'table', 'refrigerator', 'garbagebin', 
                     'picture', 'picture', 'bookshelf', 'bookshelf', 'door', 'table', 'sink', 'showercurtrain'] # We will convert AI2-THOR vocabulary to this one. Note how we're accomodating the speeling mistake in scannet data where they have 'showercurtrain' instead of 'showercurtain'.
                     
        items_transferred_directly = ['bathtub', 'bed', 'cabinet', 'chair', 'table', 'desk', 'sink', 'sofa', 'toilet', 'window'] # These items are the same in our vocabulary and AI2-THOR.
        items_to_remove = ['aluminumfoil', 'alarmclock', 'apple', 'baseballbat', 'basketball', 'candle', 'cd', 'cellphone', 'cloth',
                     'coffeemachine', 'creditcard', 'cup', 'desklamp', 'garbagebag', 'handtowel', 'handtowelholder',
                     'houseplant', 'kettle', 'keychain', 'knife', 'ladle', 'laptop', 'laundryhamper',
                     'lettuce', 'lightswitch', 'microwave', 'mirror', 'mug', 'newspaper', 'ottoman', 'pan', 'papertowelroll', 'pen', 'pencil', 'peppershaker', 'pillow',
                     'plate', 'plunger', 'pot', 'potato', 'remotecontrol', 'roomdecor',
                     'safe', 'saltshaker', 'scrubbrush', 'showerglass', 'showerhead', 'soapbar', 'soapbottle', 'spatula', 'spoon', 'spraybottle', 'statue',
                     'stool', 'stoveburner', 'stoveknob', 'tabletopdecor', 'teddybear',
                     'television', 'tennisracket', 'tissuebox', 'toaster', 'toiletpaper',
                     'toiletpaperhanger', 'tomato', 'towel', 'towelholder', 'tvstand',
                     'vacuumcleaner', 'vase', 'watch', 'wateringcan', 'winebottle', 'blinds', 'book', 'boots', 'bottle', 'bowl', 'box',
                     'bread', 'butterknife', 'dishsponge', 'dogbed', 'drawer',
                     'dresser', 'dumbbell', 'egg', 'faucet', 'floor', 'floorlamp', 'footstool',
                     'fork'] # These items we will need to remove from AI2-THOR provided set because our model can't detect them.
                     
        room_types = ['bedroom', 'bathroom', 'living room', 'kitchen'] # With the detectable models that we have, we'll be able to detect these rooms. There is nothing to detect for kitchen, so dropping that from AI2-THOR room labels.

        self.convert_ai2_thor_training_data(gen_flavour, room_types, items_to_remove, items_to_convert, converting_to)
        
    elif gen_flavour == ModelType.HYBRID_AT_12:
        file = open("features_for_each_label_AI2_THOR_12.pkl",'rb')
        features_for_each_label_ai2thor = pickle.load(file)
        file.close()

        file = open("labels_shuffled_AI2_THOR_12.pkl",'rb')
        labels_shuffled_ai2thor = pickle.load(file)
        file.close()
        
        file = open("features_for_each_label_FEATURES_12.pkl",'rb')
        features_for_each_label_gen_scannet = pickle.load(file)
        file.close()

        file = open("labels_shuffled_FEATURES_12.pkl",'rb')
        labels_shuffled_gen_scannet = pickle.load(file)
        file.close()
        
        combined_labels = labels_shuffled_ai2thor + labels_shuffled_gen_scannet
        combined_features = features_for_each_label_ai2thor + features_for_each_label_gen_scannet
        #print("Storing: " + "labels_shuffled_" + gen_flavour.name + ".pkl")
        pickle.dump(combined_labels, open("labels_shuffled_" + gen_flavour.name + ".pkl", "wb"))
        pickle.dump(combined_features, open("features_for_each_label_" + gen_flavour.name + ".pkl", "wb"))
        print("Stored " + str(len(combined_labels)) + " samples for " + gen_flavour.name)
    elif gen_flavour == ModelType.HYBRID_AT_18:
        file = open("features_for_each_label_AI2_THOR_18.pkl",'rb')
        features_for_each_label_ai2thor = pickle.load(file)
        file.close()

        file = open("labels_shuffled_AI2_THOR_18.pkl",'rb')
        labels_shuffled_ai2thor = pickle.load(file)
        file.close()
        
        file = open("features_for_each_label_FEATURES_18.pkl",'rb')
        features_for_each_label_gen_scannet = pickle.load(file)
        file.close()

        file = open("labels_shuffled_FEATURES_18.pkl",'rb')
        labels_shuffled_gen_scannet = pickle.load(file)
        file.close()
        
        combined_labels = labels_shuffled_ai2thor + labels_shuffled_gen_scannet
        combined_features = features_for_each_label_ai2thor + features_for_each_label_gen_scannet
        #print("Storing: " + "labels_shuffled_" + gen_flavour.name + ".pkl")
        pickle.dump(combined_labels, open("labels_shuffled_" + gen_flavour.name + ".pkl", "wb"))
        pickle.dump(combined_features, open("features_for_each_label_" + gen_flavour.name + ".pkl", "wb"))
        print("Stored " + str(len(combined_labels)) + " samples for " + gen_flavour.name)
    else:
        raise ValueError("Please select a valid generator flavour")

  ###
  # This will generate surrogate training data by taking real scenarios from AI2-THOR and then dropping the items that our models can't detect
  # and converting the ones that it can to our vocabulary.
  # This handles the cases of AI2_THOR_12 and AI2_THOR_18 model types (a.k.a. as sunrgbd and scannet adapted for AI2-THOR data).
  ###
  def convert_ai2_thor_training_data(self, gen_flavour, room_types, items_to_remove, items_to_convert, converting_to):
    labels_filtered = []
    features_for_each_label_filtered = []

    file = open("labels_shuffled_AI2_THOR.pkl", "rb")
    labels_shuffled = pickle.load(file)
    file.close()

    file = open("features_for_each_label_AI2_THOR.pkl", "rb")
    features_for_each_label = pickle.load(file)
    file.close()
    
    # go through all our labels
    for i in range(len(labels_shuffled)):
        if labels_shuffled[i] in room_types: # if the room type is detectable, then filter the features
            labels_filtered.append(labels_shuffled[i]) # remember the label
            features_for_this_label_filtered = list(set([e.lower() for e in features_for_each_label[i].split()]) - set(items_to_remove)) # remove the features we can't detect
            #print(features_for_each_label[i])
            features_for_this_label_filtered_converted = [] 
            for feature in features_for_this_label_filtered: # convert feature names from AI2-THOR vocabulary to ours
                if feature in items_to_convert: # if we have a feature that needs to be converted
                    features_for_this_label_filtered_converted.append(converting_to[items_to_convert.index(feature)]) # then convert it
                else:
                    features_for_this_label_filtered_converted.append(feature) # otherwise take the feature as is
    
            features_for_each_label_filtered.append(" ".join(str(e) for e in list(dict.fromkeys(features_for_this_label_filtered_converted)))) # remove duplicates and remember the filtered and converted feature list for this room
            
            #print(labels_shuffled[i] + " : ")
            #print(" ".join(str(e) for e in list(dict.fromkeys(features_for_this_label_filtered_converted))))
            
    pickle.dump(labels_filtered, open("labels_shuffled_" + gen_flavour.name + ".pkl", "wb"))
    pickle.dump(features_for_each_label_filtered, open("features_for_each_label_" + gen_flavour.name + ".pkl", "wb"))
    print("Stored " + str(len(labels_filtered)) + " samples for " + gen_flavour.name)

  ###
  # This will generate surrogate training data from scratch by randomising items in each room type from a set of allowed items for that room type.
  # This handles the cases of FEATURES_12 and FEATURES_18 model types (a.k.a. as sunrgbd and scannet)
  ###
  def generate_training_data(self, gen_flavour, room_types, possible_items_in_each_room):
    training_data = []

    # generate some data by randomizing potential objects in each room category
    print(self.TRAINING_DATASETS_TO_GENERATE)
    for i in range(self.TRAINING_DATASETS_TO_GENERATE):
        rt = random.randrange(0, len(room_types)) # choose room type to generate
        max_item_count_in_room = len(possible_items_in_each_room[rt][1]) # how many classes of items do we have in this type of room
        item_cnt_to_generate = random.randrange(1, max_item_count_in_room + 1) # how many items we will generate for this room
        #print(str(rt) + " " + str(max_item_count_in_room) + " " + str(item_cnt_to_generate))
        item_indexes = random.sample(range(0, max_item_count_in_room), item_cnt_to_generate) # the indexes of the generated items
        training_data.append((room_types[rt], [possible_items_in_each_room[rt][1][idx] for idx in item_indexes]))
        
        #print((room_types[rt], [possible_items_in_each_room[rt][1][idx] for idx in item_indexes]))

    # Now store everything in pickle files
    labels_fname = "labels_shuffled_" + gen_flavour.name + ".pkl"
    features_fname = "features_for_each_label_" + gen_flavour.name + ".pkl"

    labels_shuffled = []
    features_for_each_label = []
    #labels_shuffled = [room_data[0] for room_data in training_data]

    for room_data in training_data:
        labels_shuffled.append(room_data[0])
        # now we'll get the objects into a string separated by a space
        objs_in_room_as_string = ""
        for obj in room_data[1]:
            objs_in_room_as_string += obj + " "
        features_for_each_label.append(objs_in_room_as_string[:-1])

    pickle.dump(labels_shuffled, open(labels_fname, "wb"))
    pickle.dump(features_for_each_label, open(features_fname, "wb"))
    
    print("Stored " + str(len(labels_shuffled)) + " samples for " + gen_flavour.name)

    #print(labels_shuffled)
    #print(features_for_each_label)

    #pickle.dump(training_data, open("training_data.pkl", "wb"))

    #print(training_data)

# here we make a choice of which model we will generate the samples for.
#gen_flavour = ModelType.FEATURES_12 
#gen_flavour = ModelType.AI2_THOR_18
#gen_flavour = ModelType.FEATURES_18
#gen_flavour = ModelType.HYBRID_AT_18

#gen_flavour = ModelType.AI2_THOR_12
#gen_flavour = ModelType.FEATURES_12
#gen_flavour = ModelType.HYBRID_AT_12

#gen = RoomClassifierTrainingDataGenerator(gen_flavour)

def generateScannet():
    gen = RoomClassifierTrainingDataGenerator(ModelType.FEATURES_18)
    
def generateSunrgbd():
    gen = RoomClassifierTrainingDataGenerator(ModelType.FEATURES_12)
    
def generateAI2Thor_Scannet():
    gen = RoomClassifierTrainingDataGenerator(ModelType.AI2_THOR_18)
    
def generateAI2Thor_Sunrgbd():
    gen = RoomClassifierTrainingDataGenerator(ModelType.AI2_THOR_12)
    
def generateHybridAI2Thor_Sunrgbd(number_of_datasets_to_gen):
    gen = RoomClassifierTrainingDataGenerator(ModelType.AI2_THOR_12)
    gen = RoomClassifierTrainingDataGenerator(ModelType.FEATURES_12, number_of_datasets_to_gen)
    gen = RoomClassifierTrainingDataGenerator(ModelType.HYBRID_AT_12)
    
def generateHybridAI2Thor_Scannet(number_of_datasets_to_gen):
    gen = RoomClassifierTrainingDataGenerator(ModelType.AI2_THOR_18)
    gen = RoomClassifierTrainingDataGenerator(ModelType.FEATURES_18, number_of_datasets_to_gen)
    gen = RoomClassifierTrainingDataGenerator(ModelType.HYBRID_AT_18)
    

def main():
    #generateHybridAI2Thor_Scannet()
    generateScannet()

if __name__ == "__main__":
    main()
