import random
import pickle

from ModelType import ModelType

class RoomClassifierTrainingDataGenerator:
  ###
  # Initialises the training data generator and attempts to generate the required data based on the gen_flavour parameter.
  ###
  def __init__(self, gen_flavour):
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
        
        self.TRAINING_DATASETS_TO_GENERATE = 16000
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
        
        self.TRAINING_DATASETS_TO_GENERATE = 16000
        self.generate_training_data(gen_flavour, room_types, possible_items_in_each_room)
    elif gen_flavour == ModelType.AI2_THOR_12:
        detectable_items = ['bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
                       'night_stand', 'bookshelf', 'bathtub'] # these items we can detect with our framework
                       
        available_items = ['alarmclock', 'apple', 'armchair', 'baseballbat', 'basketball', 'bathtub',
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
                     'desktop', 'diningtable', 'sidetable', 'shelf'] # these items we will need to convert from AI2-THOR vocabulary to ours
        converting_to = ['chair', 'bathtub', 'table', 'desk', 'table', 'table', 'bookshelf'] # We will convert AI2-THOR vocabulary to this one
                     
        items_transferred_directly = ['bed', 'bathtub', 'chair', 'desk', 'dresser', 'sofa', 'toilet'] # These items are the same in our vocabulary and AI2-THOR. missing: night_stand
                     
        items_to_remove = ['alarmclock', 'apple', 'baseballbat', 'basketball', 'blinds', 'book', 'boots', 'bottle', 'bowl', 'box',
                     'bread', 'butterknife', 'cabinet', 'candle', 'cd', 'cellphone', 'cloth',
                     'coffeemachine', 'creditcard', 'cup', 'desklamp', 'dishsponge', 'dogbed', 'drawer', 'dumbbell', 'egg', 'faucet', 'floor', 'floorlamp', 'footstool',
                     'fork', 'fridge', 'garbagebag', 'garbagecan', 'handtowel', 'handtowelholder',
                     'houseplant', 'kettle', 'keychain', 'knife', 'ladle', 'laptop', 'laundryhamper',
                     'lettuce', 'lightswitch', 'microwave', 'mirror', 'mug', 'newspaper', 'ottoman',
                     'painting', 'pan', 'papertowelroll', 'pen', 'pencil', 'peppershaker', 'pillow',
                     'plate', 'plunger', 'poster', 'pot', 'potato', 'remotecontrol', 'roomdecor',
                     'safe', 'saltshaker', 'scrubbrush', 'shelf', 'shelvingunit', 'showercurtain',
                     'showerdoor', 'showerglass', 'showerhead', 'soapbar', 'soapbottle', 'spatula', 'spoon', 'spraybottle', 'statue',
                     'stool', 'stoveburner', 'stoveknob', 'tabletopdecor', 'teddybear',
                     'television', 'tennisracket', 'tissuebox', 'toaster', 'toiletpaper',
                     'toiletpaperhanger', 'tomato', 'towel', 'towelholder', 'tvstand',
                     'vacuumcleaner', 'vase', 'watch', 'wateringcan', 'window', 'winebottle', 'countertop', 'curtains', 'sink', 'sinkbasin'] # These items we will need to remove from AI2-THOR provided set because our model can't detect them.
                     
        room_types = ['bedroom', 'bathroom', 'living room'] # With the detectable models that we have, we'll be able to detect these rooms. There is nothing to detect for kitchen, so dropping that from AI2-THOR room labels.

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
    else:
        raise ValueError("Please select a valid generator flavour")

  def generate_training_data(self, gen_flavour, room_types, possible_items_in_each_room):
    training_data = []

    # generate some data by randomizing potential objects in each room category
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

    #print(labels_shuffled)
    #print(features_for_each_label)

    #pickle.dump(training_data, open("training_data.pkl", "wb"))

    #print(training_data)

# here we make a choice of which model we will generate the samples for.
#gen_flavour = ModelType.FEATURES_12 
gen_flavour = ModelType.AI2_THOR_12

gen = RoomClassifierTrainingDataGenerator(gen_flavour)

