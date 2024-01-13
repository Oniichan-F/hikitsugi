import random
import torchvision.transforms as T


def getStandardTransforms(translate=(0.3,0.3), rotation=(0,360), flip=0.5, fill=0):
    return T.Compose(
        [
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomHorizontalFlip(p = flip),
            T.RandomVerticalFlip(p = flip),
            T.RandomAffine(degrees=rotation, translate=translate, fill=fill),
        ]
    )

  
class CustomRotation:
    def __init__(self, degrees):
        self.degrees = degrees
        
    def __call__(self, x):
        degree = random.choice(self.degrees)
        return T.functional.rotate(x, degree)
    
def getRightAngleRotTransforms():
    return T.Compose([CustomRotation([90, 180, 270, 360])])