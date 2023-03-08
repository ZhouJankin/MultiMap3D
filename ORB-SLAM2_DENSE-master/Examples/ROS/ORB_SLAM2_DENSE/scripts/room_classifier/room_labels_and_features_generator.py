import random

# So we have these features - that's all we know how to detect.
# Let's construct a few labels (room types) and assign some of these items to them

# detectable_items = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
#                'bookshelf', 'picture', 'counter', 'desk', 'curtain',
#                'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
#                'garbagebin']

detectable_items = ['bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
               'night_stand', 'bookshelf', 'bathtub']
               
room_types = ['bedroom', 'bathroom', 'study', 'living_room']

possible_items_in_each_room = [
(room_types[0], ['bed', 'table', 'chair', 'dresser', 'night_stand', 'desk', 'bookshelf']),
(room_types[1], ['toilet', 'bathtub']),
(room_types[2], ['desk', 'chair', 'bookshelf']),
(room_types[3], ['table', 'sofa', 'chair', 'bookshelf', 'dresser', 'desk'])
]

training_data = []

# generate some data by randomizing potential objects in each room category
for i in range(120):
    rt = random.randrange(0, len(room_types)) # choose room type to generate
    max_item_count_in_room = len(possible_items_in_each_room[rt][1]) # how many classes of items do we have in this type of room
    
    item_cnt_to_generate = random.randrange(1, max_item_count_in_room + 1) # how many items we will generate for this room
    
    print(str(rt) + " " + str(max_item_count_in_room) + " " + str(item_cnt_to_generate))
    
    item_indexes = random.sample(range(0, max_item_count_in_room), item_cnt_to_generate) # the indexes of the generated items
    training_data.append((room_types[rt], [possible_items_in_each_room[rt][1][idx] for idx in item_indexes]))

print(training_data)
