from enum import Enum

###
# When we generate the training samples, we can generate them for 2 different detection engines, that we've worked on.
# One of those engines detects 12 classes of objects, the other- 18. We have to make a choice which one we will 
# generate the samples for. This enum will help with that.
###
class ModelType(Enum):
    FEATURES_12 = 12 # for the model with 12 detectable classes # Turns out that FEATURES_12 detector a.k.a. sunrgbd class actually only provides 10 classes, not 12, but I'm not changing this anymore because it's referenced in many places in code.
    FEATURES_18 = 18 # for the model with 18 detectable classes
    AI2_THOR = 100
    AI2_THOR_12 = 112 # for AI2-THOR training data with classes filtered out to only contain the 12 from the 12-sample detection engine
    AI2_THOR_18 = 118 # for AI2-THOR training data with classes filtered out to only contain the 18 from the 18-sample detection engine
