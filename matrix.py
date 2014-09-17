import math

class Vec3 (object):
    """Vec3 is a 3 dimensional vector class used for both direction vectors and 3D points.  
    Vec3's methods return new vectors (or scalar values) and do not modify the object"""
    def __init__(self, x=0,y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
    
    def length(self):
        """"Returns the Euclidean length of the vector"""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    
    def normalize(self):
        """"Returns a unit vector with the same direction as self.  normalize does not modify self."""
        length = self.length()
        if(length != 0):
            return self.scalarMult(1.0/length)

    def dot(self, other):
        """returns the dot product of self and another vector.  a dot b = |a| |b| cos(theta) where theta is the angle between vectors a and b."""
        return self.x * other.x + self.y * other.y + self.z * other.z
        
    def cross(self, other):
        """returns the cross product of self and another vector.  The returned vector is orthogonal (perpendicular) to both self and other."""
        x = -self.y * other.z + self.z * other.y
        y = -self.z * other.x + self.x * other.z
        z = -self.x * other.y + self.y * other.x
        return Vec3(x,y,z)
    
    def scalarMult(self, value):
        """returns a new vector which is the result of self multiplied by value"""
        return Vec3(self.x * value, self.y * value, self.z * value)
    
    def mult(self, b):
        return Vec3(self.x * b.x, self.y * b.y, self.z * b.z)
    
    def neg(self):
        """returns a new vector which is the result of self multiplied by -1.  This vector points in the opposite direction as self"""
        return self.scalarMult(-1);
    
    def add(self, other):
        """returns a new vector which is the sum of self and other."""
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def sub(self, other):
        """returns a new vector which is the differents of self and other."""
        return self.add(other.neg());
    
    def __repr__(self):
        """This function returns a nicely formatted string representing self.  __repr__ is called by print."""
        return "(" + str(self.x) +", " + str(self.y) +", " + str(self.z) + ")"


class Matrix (object):
	"""Matrix is a 4x4 matrix."""
	
	def __init__(self, matls = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]):
		"""__init__ expects a list containing 4 4-element lists.  Each list is a row of the matrix"""
                [
                    [self.m00, self.m01, self.m02, self.m03],
                    [self.m10, self.m11, self.m12, self.m13],
                    [self.m20, self.m21, self.m22, self.m23],
                    [self.m30, self.m31, self.m32, self.m33]
                    ] = matls
         
        def mult(self, b):
            """returns a new matrix which is the product of self and matrix b"""
            m00 = self.m00 * b.m00 + self.m01 * b.m10 + self.m02 * b.m20 + self.m03 * b.m30
            m01 = self.m00 * b.m01 + self.m01 * b.m11 + self.m02 * b.m21 + self.m03 * b.m31
            m02 = self.m00 * b.m02 + self.m01 * b.m12 + self.m02 * b.m22 + self.m03 * b.m32
            m03 = self.m00 * b.m03 + self.m01 * b.m13 + self.m02 * b.m23 + self.m03 * b.m33
            
            m10 = self.m10 * b.m00 + self.m11 * b.m10 + self.m12 * b.m20 + self.m13 * b.m30
            m11 = self.m10 * b.m01 + self.m11 * b.m11 + self.m12 * b.m21 + self.m13 * b.m31
            m12 = self.m10 * b.m02 + self.m11 * b.m12 + self.m12 * b.m22 + self.m13 * b.m32
            m13 = self.m10 * b.m03 + self.m11 * b.m13 + self.m12 * b.m23 + self.m13 * b.m33
            
            m20 = self.m20 * b.m00 + self.m21 * b.m10 + self.m22 * b.m20 + self.m23 * b.m30
            m21 = self.m20 * b.m01 + self.m21 * b.m11 + self.m22 * b.m21 + self.m23 * b.m31
            m22 = self.m20 * b.m02 + self.m21 * b.m12 + self.m22 * b.m22 + self.m23 * b.m32
            m23 = self.m20 * b.m03 + self.m21 * b.m13 + self.m22 * b.m23 + self.m23 * b.m33
            
            m30 = self.m30 * b.m00 + self.m31 * b.m10 + self.m32 * b.m20 + self.m33 * b.m30
            m31 = self.m30 * b.m01 + self.m31 * b.m11 + self.m32 * b.m21 + self.m33 * b.m31
            m32 = self.m30 * b.m02 + self.m31 * b.m12 + self.m32 * b.m22 + self.m33 * b.m32
            m33 = self.m30 * b.m03 + self.m31 * b.m13 + self.m32 * b.m23 + self.m33 * b.m33
        
            return Matrix([[m00,m01,m02,m03],
                       [m10,m11,m12,m13],
                       [m20,m21,m22,m23],    
                       [m30,m31,m32,m33]])
        def __repr__(self):
            """This function returns a nicely formatted string representing self.  __repr__ is called by print."""
            return "[[ " + str(self.m00) +","+ str(self.m01)+","+ str(self.m02)+","+ str(self.m03)+"],["+ \
                str( self.m10)+","+str(self.m11)+","+str(self.m12)+","+ str(self.m13)+"],["+ \
                str(self.m20)+","+str(self.m21)+","+ str(self.m22)+","+ str(self.m23)+"],["+ \
                str(self.m30)+","+str(self.m31)+","+ str(self.m32)+","+ str(self.m33)+"]]"
        
        
        def vecMult(self, vec, w=1):
            """Multiplies (transforms) a Vec3.  Use w=1 for standard point transformatons, use w=0 for transformation of direction vectors. 
            This method assumes affine transformation so we do not care about last row of matrix
            Returns a Vec3"""
            x = vec.x * self.m00 + vec.y * self.m01 + vec.z * self.m02 + w * self.m03
            y = vec.x * self.m10 + vec.y * self.m11 + vec.z * self.m12 + w * self.m13
            z = vec.x * self.m20 + vec.y * self.m21 + vec.z * self.m22 + w * self.m23
            return Vec3(x,y,z)
    
        def  determinant(self):
            """returns the determinant of the matrix"""
            return self.m03*self.m12*self.m21*self.m30 - self.m02*self.m13*self.m21*self.m30 - self.m03*self.m11*self.m22*self.m30 + self.m01*self.m13*self.m22*self.m30+ \
                self.m02*self.m11*self.m23*self.m30 - self.m01*self.m12*self.m23*self.m30 - self.m03*self.m12*self.m20*self.m31 + self.m02*self.m13*self.m20*self.m31+\
                self.m03*self.m10*self.m22*self.m31 - self.m00*self.m13*self.m22*self.m31 - self.m02*self.m10*self.m23*self.m31 + self.m00*self.m12*self.m23*self.m31+\
                self.m03*self.m11*self.m20*self.m32 - self.m01*self.m13*self.m20*self.m32 - self.m03*self.m10*self.m21*self.m32 + self.m00*self.m13*self.m21*self.m32+\
                self.m01*self.m10*self.m23*self.m32 - self.m00*self.m11*self.m23*self.m32 - self.m02*self.m11*self.m20*self.m33 + self.m01*self.m12*self.m20*self.m33+\
                self.m02*self.m10*self.m21*self.m33 - self.m00*self.m12*self.m21*self.m33 - self.m01*self.m10*self.m22*self.m33 + self.m00*self.m11*self.m22*self.m33
        
        def inverse(self): 
            """returns the inverse of the matrix.  If not inverse exists returns none"""
            d = self.determinant();
            if(d==0):
                return None
            d = 1.0/d
            m00 = (self.m12*self.m23*self.m31 - self.m13*self.m22*self.m31 + self.m13*self.m21*self.m32 - self.m11*self.m23*self.m32 - self.m12*self.m21*self.m33 + self.m11*self.m22*self.m33)*d
            m01 = (self.m03*self.m22*self.m31 - self.m02*self.m23*self.m31 - self.m03*self.m21*self.m32 + self.m01*self.m23*self.m32 + self.m02*self.m21*self.m33 - self.m01*self.m22*self.m33)*d
            m02 = (self.m02*self.m13*self.m31 - self.m03*self.m12*self.m31 + self.m03*self.m11*self.m32 - self.m01*self.m13*self.m32 - self.m02*self.m11*self.m33 + self.m01*self.m12*self.m33)*d
            m03 = (self.m03*self.m12*self.m21 - self.m02*self.m13*self.m21 - self.m03*self.m11*self.m22 + self.m01*self.m13*self.m22 + self.m02*self.m11*self.m23 - self.m01*self.m12*self.m23)*d
            m10 = (self.m13*self.m22*self.m30 - self.m12*self.m23*self.m30 - self.m13*self.m20*self.m32 + self.m10*self.m23*self.m32 + self.m12*self.m20*self.m33 - self.m10*self.m22*self.m33)*d
            m11 = (self.m02*self.m23*self.m30 - self.m03*self.m22*self.m30 + self.m03*self.m20*self.m32 - self.m00*self.m23*self.m32 - self.m02*self.m20*self.m33 + self.m00*self.m22*self.m33)*d
            m12 = (self.m03*self.m12*self.m30 - self.m02*self.m13*self.m30 - self.m03*self.m10*self.m32 + self.m00*self.m13*self.m32 + self.m02*self.m10*self.m33 - self.m00*self.m12*self.m33)*d
            m13 = (self.m02*self.m13*self.m20 - self.m03*self.m12*self.m20 + self.m03*self.m10*self.m22 - self.m00*self.m13*self.m22 - self.m02*self.m10*self.m23 + self.m00*self.m12*self.m23)*d
            m20 = (self.m11*self.m23*self.m30 - self.m13*self.m21*self.m30 + self.m13*self.m20*self.m31 - self.m10*self.m23*self.m31 - self.m11*self.m20*self.m33 + self.m10*self.m21*self.m33)*d
            m21 = (self.m03*self.m21*self.m30 - self.m01*self.m23*self.m30 - self.m03*self.m20*self.m31 + self.m00*self.m23*self.m31 + self.m01*self.m20*self.m33 - self.m00*self.m21*self.m33)*d
            m22 = (self.m01*self.m13*self.m30 - self.m03*self.m11*self.m30 + self.m03*self.m10*self.m31 - self.m00*self.m13*self.m31 - self.m01*self.m10*self.m33 + self.m00*self.m11*self.m33)*d
            m23 = (self.m03*self.m11*self.m20 - self.m01*self.m13*self.m20 - self.m03*self.m10*self.m21 + self.m00*self.m13*self.m21 + self.m01*self.m10*self.m23 - self.m00*self.m11*self.m23)*d
            m30 = (self.m12*self.m21*self.m30 - self.m11*self.m22*self.m30 - self.m12*self.m20*self.m31 + self.m10*self.m22*self.m31 + self.m11*self.m20*self.m32 - self.m10*self.m21*self.m32)*d
            m31 = (self.m01*self.m22*self.m30 - self.m02*self.m21*self.m30 + self.m02*self.m20*self.m31 - self.m00*self.m22*self.m31 - self.m01*self.m20*self.m32 + self.m00*self.m21*self.m32)*d
            m32 = (self.m02*self.m11*self.m30 - self.m01*self.m12*self.m30 - self.m02*self.m10*self.m31 + self.m00*self.m12*self.m31 + self.m01*self.m10*self.m32 - self.m00*self.m11*self.m32)*d
            m33 = (self.m01*self.m12*self.m20 - self.m02*self.m11*self.m20 + self.m02*self.m10*self.m21 - self.m00*self.m12*self.m21 - self.m01*self.m10*self.m22 + self.m00*self.m11*self.m22)*d
            return Matrix([[m00,m01,m02,m03],[m10,m11,m12,m13],[m20,m21,m22,m23],[m30,m31,m32,m33]])
        def transpose(self):
            """returns the transpose of the matrix"""
            return Matrix([[self.m00, self.m10, self.m20, self.m30],[self.m01, self.m11, self.m21, self.m31],[self.m02, self.m12, self.m22, self.m32], [self.m03, self.m13, self.m23, self.m33]])
		
def test():
    a = Matrix([[2,0,0,2],[0,2,0,7],[0,0,2,4],[4,0,7,1]])
    print a.mult(a)
    b = a.inverse()
    print b
    c = a.mult(b)
    print c
    t = Matrix([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]).transpose()
    print t


        
