from enum import Enum

###
# When we generate the training samples, we can generate them for 2 different detection engines, that we've worked on.
# One of those engines detects 12 classes of objects, the other- 18. We have to make a choice which one we will 
# generate the samples for. This enum will help with that.
###
class ModelType(Enum):
    FEATURES_12 = 12 # For sunrgbd detector (10 features) - generated rooms and features from available data. # Turns out that FEATURES_12 detector a.k.a. sunrgbd class actually only provides 10 classes, not 12, but I'm not changing this anymore because it's referenced in many places in code.
    FEATURES_18 = 18 # For scannet detector (18 features) - generated rooms and features from available data.    
    AI2_THOR = 100 # For the original AI2-THOR data extracted from the simulation
    AI2_THOR_12 = 112 # For AI2-THOR vocabulary filtered and translated to sunrgbd data and then AI2-THOR data used with sunrgbd features
    AI2_THOR_18 = 118 # For AI2-THOR vocabulary filtered and translated to scannet data and then AI2-THOR data used with scannet features
    HYBRID_AT_18 = 218 # For AI2-THOR vocabulary filtered and translated to scannet data and then AI2-THOR data used with scannet features and then scannet generated cases added on top of the ones that came from AI2-THOR
    HYBRID_AT_12 = 212 # For AI2-THOR vocabulary filtered and translated to sunrgbd data and then AI2-THOR data used with sunrgbd features and then sunrgbd generated cases added on top of the ones that came from AI2-THOR
