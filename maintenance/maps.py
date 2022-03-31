# Maps for columns
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

mapDict['designLoad'] = {
                         1:'H10',
                         2:'H15',
                         3:'HS15',
                         4:'H20',
                         5:'HS20',
                         6:'HS20Mod',
                         7:'Pedestrian',
                         8:'Railroad',
                         9:'HS25',
                       }

mapDict['deckStructureType'] = {
                         1:'ConcreteCastInPlace',
                         2:'ConcretePrecastPanels',
                         3:'OpenGrating',
                         4:'CloseGrating',
                         5:'SteelPlate',
                         6:'CorrugatedSteel',
                         7:'Aluminum',
                         8:'Wood',
                         9:'Other',
                        }


mapDict['typeOfDesign'] = {
                         1:'Slab',
                         2:'StringerMultiBeam',
                         3:'GirderAndFloor',
                         4:'TeeBeam',
                         5:'BoxBeamMultiple',
                         6:'BoxBeamSingle',
                         7:'Frame',
                         8:'Orthotropic',
                         9:'TrussDeck',
                         10:'TrussThru',
                         11:'ArchDeck',
                         12:'ArchThru',
                         13:'Suspension',
                         14:'StayedGirder',
                         15:'MovableLift',
                         16:'MovableBascule',
                         17:'MovableSwing',
                         18:'Tunnel',
                         19:'Culvert',
                         20:'MixedTypes',
                         21:'SegmentalBoxGirder',
                         22:'ChannelBeam',
                         0:'Other',
                        }
