import random
import pickle

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

training_data = []

TRAINING_DATASETS_TO_GENERATE = 12000

# generate some data by randomizing potential objects in each room category
for i in range(TRAINING_DATASETS_TO_GENERATE):
    rt = random.randrange(0, len(room_types)) # choose room type to generate
    max_item_count_in_room = len(possible_items_in_each_room[rt][1]) # how many classes of items do we have in this type of room
    item_cnt_to_generate = random.randrange(1, max_item_count_in_room + 1) # how many items we will generate for this room
    #print(str(rt) + " " + str(max_item_count_in_room) + " " + str(item_cnt_to_generate))
    item_indexes = random.sample(range(0, max_item_count_in_room), item_cnt_to_generate) # the indexes of the generated items
    training_data.append((room_types[rt], [possible_items_in_each_room[rt][1][idx] for idx in item_indexes]))
    
    #print((room_types[rt], [possible_items_in_each_room[rt][1][idx] for idx in item_indexes]))

# Now store everything in pickle files
labels_fname = "labels_shuffled.pkl"
features_fname = "features_for_each_label.pkl"

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
