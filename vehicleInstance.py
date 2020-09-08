'''
vehicleInstance class
'''
class vehicleInstance:
    def __init__(self, subImage_id=0):
        self.subImage_id = subImage_id
        self.reid_id = 0
        self.mask = None
        self.embed_feature = None
        self.bbox = None
        self.crop_image = None
        self.plate = ""
        self.directionVector = None