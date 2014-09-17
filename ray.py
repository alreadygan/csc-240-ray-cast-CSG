from matrix import Vec3
'''
Created on Oct 10, 2011

@author: eitan
'''



class Point2D (object):
    """Point2D is used to represent 2D points.  The expected use of this class is for screen coordinates pass into Camera.generateRay."""
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    


class Ray (object):
    '''
    classdocs
    '''

    def __init__(self, dir = Vec3(1,1,1), orig= Vec3(0,0,0) ):
        self.direction = dir
        self.origin = orig
        '''
        Constructor
        '''
        pass
    
    def getOrigin(self): 
        return self.origin
    
    def getDirection(self): 
        return self.direction
    
    def pointAtParameter(self, t): 
        return self.origin.add(self.direction.scalarMult(t))
    def __repr__(self):
        return "[" + str(self.origin) + "-->" + str(self.direction) + "]"
    
class Hit (object):
    def __init__(self, t=float('inf'), color=(0,0,0),  normal = Vec3()):
        self.t = t
        self.color= color
        self.normal = normal
    
    def getT(self):
        return self.t

    def getColor(self):
        return self.color
    
    def getNormal(self):
        return self.normal
    def __repr__(self):
        return "Hit(" + str(self.t) + ")"
        
