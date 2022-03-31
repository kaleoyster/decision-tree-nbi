# Material map
from collections import defaultdict

mapDict = defaultdict()
mapDict['material'] = {
                         1:'Concrete',
                         2:'ConcreteContinuous',
                         3:'Steel',
                         4:'SteelConitnuous',
                         5:'PrestreesedConcrete',
                         6:'PrestreesedConcreteContinuous',
                         7:'Wood',
                         8:'Masonry',
                         9:'Aluminum',
                         0:'Other',
                    }

mapDict['toll'] = {
                         1:'TollBridge',
                         2:'OnTollBridge',
                         3:'OnFreeRoad',
                         4:'OnInterstateToll',
                         5:'TollBridgeSegementUnder',
                    }
