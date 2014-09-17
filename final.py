"""
final.py
Author: Adriane Gan
Ray Caster with Constructive Solid Geometry
Final Project for CSC240 at Smith College
Fall 2013
"""
import matrix as m
import ray as r
import math
import raster

#-------------------------------------------------------
# Camera Section
#-------------------------------------------------------

class Camera (object):
    """Base class for cameras.  Camera's are responsible for generating Rays (which are used by generateImage)."""
    def __init__(self, location, up, dir, width, height):
        self.location = location
        self.up = up.normalize()
        self.dir = dir.normalize()
        self.width = width
        self.height = height
        self.horz = self.up.cross(self.dir).normalize() # used by all Cameras so might as well calc here

    def generateRay(self, point):
        """generateRay takes a Point2D and returns a ray.  It expects x and y to be between 0 and 1"""
        pass

    def screenToPlaneLocation(self, screenCoord):
        """Takes a Point2D returns a 3D location as a Vec3.  The Point2D should have x and y to be between 0 and 1.
        The returned Vec3 is x,y mapped onto the projection plain."""
        return self.location.add(self.up.scalarMult(screenCoord.y* self.height)).add(self.horz.scalarMult(screenCoord.x*self.width))

class OrthographicCamera (Camera):
    def __init__(self, location, up, dir, width, height):
        """Center, dir, and up define the projection plane.  Width and height define its size"""
        super(OrthographicCamera, self).__init__(location, up, dir, width, height)


    def generateRay(self, screenCoord):
        """
        generateRay takes a Point2D and returns a ray  It expects x and y to be between 0 and 1.
        The origin on the ray corresponds to x,y mapped onto the projection plain.
        The direction of the ray is defined by the camera's initializer
        """
        return r.Ray(self.dir, super(OrthographicCamera, self).screenToPlaneLocation(screenCoord));


class PerspectiveCamera (Camera):
    def __init__(self, location, up, dir, width, height, eye):
        """Center, dir, and up define the projection plane.  Width and height define its size"""
        super(PerspectiveCamera, self).__init__(location, up, dir, width, height)
        self.eye = eye


    def generateRay(self, screenCoord):
        """
        generateRay takes a Point2D and returns a ray  It expects x and y to be between 0 and 1.
        The origin on the ray corresponds to x,y mapped onto the projection plain.
        The direction of the ray is defined by the camera's initializer
        """
        orig = super(PerspectiveCamera, self).screenToPlaneLocation(screenCoord)
        return r.Ray(orig.sub(self.eye).normalize(), self.eye);

#-------------------------------------------------------
# Object3D base and child classes
#-------------------------------------------------------

class Object3D (object):
    def __init__(self, color = r.Vec3(0.1,0.1,0.1)):
        """___init___ takes a optional color.  If no color is provided the Object is dark grey"""
        self.color = color

    def intersect(self, ray, hit, tmin, tmax):
        """intersect takes a ray, hit, tmin, and tmax.
        intersect returns a new Hit object that represents the closest intersection between tmin and tmax of the ray and the object
        if that intersection is closer than hit.
        hit is returned if no intersection is found between tmin and tmax or
        the intersection is not closer than hit."""
        pass

class Sphere (Object3D):
    def __init__(self, center, radius, color):
        """Takes a center, radius, and color"""
        super(Sphere, self).__init__(color)
        self.center = center
        self.radius = radius

    def intersect(self, ray, hit, tmin, tmax):
        """intersect takes a ray, hit, tmin, and tmax.
        intersect returns a new Hit object that represents the closest intersection between tmin and tmax of the ray and the object
        if that intersection is closer than hit.
        hit is returned if no intersection is found between tmin and tmax or
        the intersection is not closer than hit."""
        ro = ray.getOrigin().sub(self.center);
        b = 2 * ray.getDirection().dot(ro)
        c = ro.dot(ro) - self.radius * self.radius;
        d = b*b-4*c
        if(d<0): # no real positive root
            return hit # no new hit
        d = math.sqrt(d)
        t1 = (-b + d) * .5
        t2 = (-b - d) * .5

        if(t2<t1): # make sure t1<t2
            t1,t2 = t2,t1

        #make t the smallest valid t value
        if  tmin<= t1<=tmax:
            t = t1
        elif  tmin<= t2<=tmax:
            t = t2
        else: # no valid t
            return hit;

        if(t < hit.getT()) :  # closer than current t
            # VALID HIT
            norm = ray.pointAtParameter(t).sub(self.center)
            return r.Hit(t, self.color, norm)
        else:
            return hit

    def intersectAll(self, ray, tmin, tmax):
        ''' used by CSG - needs all hits for an obj, not just closest '''
        ro = ray.getOrigin().sub(self.center);
        b = 2 * ray.getDirection().dot(ro)
        c = ro.dot(ro) - self.radius * self.radius;
        d = b*b-4*c
        if(d<0): # no real positive root
            return r.Hit() # no new hit
        d = math.sqrt(d)
        t1 = (-b + d) * .5
        t2 = (-b - d) * .5

        if(t2<t1): # make sure t1<t2
            t1,t2 = t2,t1

        norm1 = ray.pointAtParameter(t1).sub(self.center)
        norm2 = ray.pointAtParameter(t2).sub(self.center)

        return((r.Hit(t1,self.color,norm1),r.Hit(t2,self.color,norm2)))

    def __repr__(self):
        """"__repr__ returns a string representation of Sphere and is used by print."""
        return "Sphere(" + str(self.center)+", "+str(self.radius) + ")"

class Plane(Object3D):
    def __init__(self, normal, d, color):
        super(Plane,self).__init__(color)
        self.d = d
        self.normal = normal

    def intersect(self,ray,hit,tmin,tmax):
        vd = self.normal.dot(ray.getDirection())
        if vd >= 0:
            return hit
        else:
            vo = (self.normal.dot(ray.getOrigin()) + self.d)
            t = vo / vd
            return r.Hit(t,self.color,self.normal)


class Triangle (Object3D):
    def __init__(self,p1,p2,p3,color):
        super(Triangle,self).__init__(color)
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

    # reuse intersect method for plane: remove triangle bounds condition
    def intersect(self, ray, hit, tmin, tmax):
        # move this section to initializer
        v1 = self.p2.sub(self.p1)
        v2 = self.p3.sub(self.p1)
        triNorm = v1.cross(v2).normalize()
        p = ray.direction.cross(v2)
        q = (ray.origin.sub(self.p1)).cross(v1)

        # denominator used to find scalar for Barycentric coord
        den = p.dot(v1)
        if den == 0:
            return hit
        else:
            # find scalar, beta, and gamma
            s = 1/float(den)
            bet = s * p.dot(ray.origin.sub(self.p1))
            gam = s * q.dot(ray.direction)

        # check barycentric coords for inside-triangle validity
        if (bet < 0) | (gam < 0) | (bet + gam > 1):
            return hit
        else:
            # calculate t and return new hit if inside triangle
            t = s * q.dot(v2)
            return r.Hit(t,self.color,triNorm)

class Cube(Object3D):
    def __init__(self, center, side, color):
        self.center = center
        self.side = side
        self.color = color

    def intersect(self,ray,hit,tmin,tmax):
        hits = self.intersectAll(ray,tmin,tmax)

        hits = sorted(hits, key=lambda hit:hit.t)
       
        return hits[0]


    def intersectAll(self,ray,tmin,tmax):
        l = float(self.side)/2
        p1 = self.center.add(r.Vec3( l, l, l))
        p2 = self.center.add(r.Vec3( l, l,-l))
        p3 = self.center.add(r.Vec3( l,-l,-l))
        p4 = self.center.add(r.Vec3(-l, l, l))
        p5 = self.center.add(r.Vec3(-l,-l, l))
        p6 = self.center.add(r.Vec3(-l,-l,-l))
        
        hits = []
        # top
        hits.append(self.intersectSide(ray,tmin,tmax,(p1,p4,p2)))
        # right side
        hits.append(self.intersectSide(ray,tmin,tmax,(p2,p3,p1)))
        # back
        hits.append(self.intersectSide(ray,tmin,tmax,(p3,p6,p2)))
        # front
        hits.append(self.intersectSide(ray,tmin,tmax,(p4,p1,p5)))
        # left side
        hits.append(self.intersectSide(ray,tmin,tmax,(p5,p6,p4)))
        # bottom
        hits.append(self.intersectSide(ray,tmin,tmax,(p6,p3,p5)))

        return hits

    def intersectSide(self, ray, tmin, tmax,points):
        v1 = points[1].sub(points[0])
        v2 = points[2].sub(points[0])
        norm = v1.cross(v2).normalize()
        p = ray.direction.cross(v2)
        q = (ray.origin.sub(points[0])).cross(v1)

        # denominator used to find scalar for Barycentric coord
        den = p.dot(v1)
        if den == 0:
            return r.Hit()
        else:
            # find scalar, beta, and gamma
            s = 1/float(den)
            bet = s * p.dot(ray.origin.sub(points[0]))
            gam = s * q.dot(ray.direction)

        # check barycentric coords for inside-square validity
        if (bet < 0) | (gam < 0) | (bet > 1) | (gam > 1):
            return r.Hit()
        else:
            # calculate t and return new hit if inside square
            t = s * q.dot(v2)
            if (tmin < t) & (t < tmax):
                return r.Hit(t,self.color,norm)
            else:
                return r.Hit()
        
        
#-------------------------------------------------------
# CSG
#-------------------------------------------------------

class Operation(object):
    '''Enumerator for CSG operations'''
    UNION = 0
    INTERSECTION = 1
    DIFFERENCE = 2

class CSG(Object3D):
    def __init__(self, object1, object2, operation):
        self.obj1 = object1
        self.obj2 = object2
        self.op = operation

    def intersectAll(self, ray, tmin,tmax):
        g = Group()
        g.addObject(self.obj1)
        g.addObject(self.obj2)
        hits = g.intersectAll(ray,tmin,tmax)
        # should sort (hit/obj) pairs by hit.t values
        hits = sorted(hits, key=lambda extendedHit:extendedHit[0].t)


        if hits==[]:
            return r.Hit()

        else:
            if self.op == Operation.UNION:
                return(hits)

            elif self.op == Operation.INTERSECTION:
                # intersection: space belonging to both objects

                ints = self.intersectionHits(hits,tmin,tmax)

                ints = sorted(ints, key=lambda hit:hit.t)
                return ints

            elif self.op == Operation.DIFFERENCE:

                difs = self.differenceHits(hits,tmin,tmax)

                difs = sorted(difs, key=lambda hit:hit.t)
                if difs==[]:
                    return r.Hit()
                else:
                    return difs


    def intersect(self, ray, hit, tmin, tmax):
        g = Group()
        g.addObject(self.obj1)
        g.addObject(self.obj2)
        closestHit = r.Hit()

        hits = g.intersectAll(ray,tmin,tmax)
        # intersectAll returns a list of tuples (hit, object)

        if hits==[]:
            return closestHit

        # should sort (hit/obj) pairs by hit.t values
        hits = sorted(hits, key=lambda extendedHit:extendedHit[0].t)

        if self.op == Operation.UNION:
            # since union is all hits for both objects,
            # use the closest hit (smallest t in hits)
            for h in hits:
                if (tmin <= h.t) & (h.t <= tmax):
                    closestHit = h

        elif self.op == Operation.INTERSECTION:
            # intersection: space belonging to both objects

            ints = self.intersectionHits(hits,tmin,tmax)

            for h in ints:
                if h.t < closestHit.t:
                    closestHit = h

        elif self.op == Operation.DIFFERENCE:
            # difference: space belonging to one object but not the other
            difs = self.differenceHits(hits,tmin,tmax)

            for h in difs:
                if h.t < closestHit.t:
                    closestHit = h

        else:
            print("Invalid CSG operation.")

        return closestHit

    def intersectionHits(self,hits,tmin,tmax):
        '''
        This code is used in both intersect and intersectAll.
        It is defined as a function to avoid duplication of code.
        takes a list of (hit/object) tuples
        returns a list of hits
        '''
        inObj1 = False
        inObj2 = False
        ints = []

        for h in hits:
            hobj = h[1]
            if (hobj == self.obj1):
                inObj1 = not(inObj1)
            elif (hobj == self.obj2):
                inObj2 = not(inObj2)
            else:
                print("Something's wrong with CSG intersection!")

            if (inObj1) & (inObj2) & (tmin < h[0].t) & (h[0].t < tmax):
                ints.append(h[0])

        return ints

    def differenceHits(self,hits,tmin,tmax):
        '''
        This code is used in both intersect and intersectAll.
        It is defined as a function to avoid duplication of code.
        takes a list of (hit/object) tuples
        returns a list of hits
        '''
        inObj1 = False
        inObj2 = False
        difs = []

        for h in hits:

            hobj = h[1]
            if (hobj == self.obj1):
                inObj1 = not(inObj1)
            elif (hobj == self.obj2):
                inObj2 = not(inObj2)
            else:
                # if/elif/else used instead of if/else because I felt like it (there is no actual else case)
                print("Something's wrong with CSG difference!")

            if (inObj1) & (inObj2==False) & (tmin < h[0].t) & (h[0].t < tmax):
                # get the solid geometry from shape being exited
                if (hobj == self.obj2):
                    difs.append(r.Hit(h[0].t, h[0].color, h[0].normal.neg()))
                else:
                    difs.append(h[0])

        return difs
    
#-------------------------------------------------------
# Transformation class
#-------------------------------------------------------

class Transformation (Object3D):

    def __init__(self, obj):
        self.obj = obj
        self.mat = m.Matrix()

    def setTransform(self, m2):
        self.mat = m2

    def addTransform(self, m2):
        self.mat = self.mat.mult(m2)

    def intersect(self,ray,hit,tmin,tmax):
        # transform ray by inverse matrix
        inv = self.mat.inverse()
        tro = inv.vecMult(ray.getOrigin())          # tro = Transformed Ray Origin
        trd = inv.vecMult(ray.getDirection(), 0)    # trd = Transformed Ray Direction

        # normalize transformed direction
        trdn = trd.normalize()

        # define transformed ray as a ray
        tr = r.Ray(trdn,tro)
        s = trd.length()

        # Get object hit with transformed ray
        h = self.obj.intersect(tr,hit,tmin*s,tmax*s)

        if h.t == float('inf'):
            return hit
        else:
            # transform the normal by transposed inverse matrix
            invtr = inv.transpose()
            tnor = invtr.vecMult(h.normal).normalize()

            t = h.t/float(s)
            return r.Hit(t,h.color,tnor)

    def intersectAll(self,ray,tmin,tmax):
        h = self.intersect(ray,r.Hit(),tmin,tmax)
        return h

#-------------------------------------------------------
# Group
#-------------------------------------------------------

class Group (Object3D):

    def __init__(self):
        super(Group, self).__init__()
        self.objs = []

    def addObject(self,obj):
        self.objs.append(obj)

    def intersect(self, ray, tmin, tmax):
        closest_hit = r.Hit()
        for o in self.objs:
            h = o.intersect(ray, closest_hit,tmin, tmax)
            if (h.t < closest_hit.t) & (tmin < h.t) & (h.t < tmax):
                closest_hit = h
        return closest_hit

    def intersectAll(self, ray, tmin, tmax):
        '''
        used by CSG
        returns a list of tuples associating hits with their objects
        '''
        hits = []
        for o in self.objs:
            h = o.intersectAll(ray, tmin, tmax)
            try:
                for i in range(len(h)):
                    hits.append((h[i],o))

            # may not get a tuple from intersectAll
            except TypeError:
                    hits.append((h,o))
        return hits

#-------------------------------------------------------
# Lights
#-------------------------------------------------------

class Light(object):
    def __init__(self,color):
        self.color = color

    def getIntensity(intersectionPoint, surfaceNormal):
        pass

class AmbientLight(Light):
    def __init__(self,color):
        super(AmbientLight, self).__init__(color)

    def getIntensity(self,intersectionPoint, surfaceNormal):
        return self.color

class DirectionalLight(Light):
    def __init__(self, direction, color):
        super(DirectionalLight, self).__init__(color)
        self.direction = direction

    def getIntensity(self,intersectionPoint, surfaceNormal):
        dotp = surfaceNormal.dot(self.direction)
        if dotp > 0:
            return self.color.scalarMult(dotp)
        else:
            return r.Vec3(0,0,0)

#-------------------------------------------------------
# Rendering
#-------------------------------------------------------

class RenderMode (object):
    COLOR = 0
    DISTANCE = 1
    DIFFUSE = 2
    SHADOWS = 3

def generateImage(width, height, camera, group, dist_min, dist_max, lights, renderMode,bgc=r.Vec3(1,1,1)):
    #create Raster
    ra = raster.Raster(width, height)
    #loop over all pixels
    for x in range(0,width):
        for y in range(0,height):

            #create Ray
            rayX = float(x)/width
            rayY = float(height-y)/height
            #ray = camera.screenToPlaneLocation(r.Point2D(rayX,rayY))
            ray = camera.generateRay(r.Point2D(rayX,rayY))

            #find closest hit
            h = group.intersect(ray, dist_min, dist_max)
            
            #in case no hit is found
            if h.t != float('inf'):
                #set pixel color based on render mode
                if renderMode == RenderMode.COLOR:
                    color = h.color

                elif renderMode == RenderMode.DISTANCE:
                    gray = (dist_max-h.t)/(dist_max - dist_min)
                    color = r.Vec3(gray,gray,gray)

                elif renderMode == RenderMode.DIFFUSE:
                    lsum = r.Vec3(0,0,0)
                    intersectionPoint = ray.pointAtParameter(h.t)
                    surfaceNormal = h.getNormal().normalize()
                    for l in lights:
                        lco = l.getIntensity(intersectionPoint,surfaceNormal)
                        lsum.x += h.color.x * lco.x
                        lsum.y += h.color.y * lco.y
                        lsum.z += h.color.z * lco.z
                    # Check lsum values (cap at 1)
                    if lsum.x > 1:lsum.x = 1
                    if lsum.y > 1:lsum.y = 1
                    if lsum.z > 1:lsum.z = 1

                    color = lsum


                elif renderMode == RenderMode.SHADOWS:
                    lsum = r.Vec3(0,0,0)
                    intersectionPoint = ray.pointAtParameter(h.t)
                    surfaceNormal = h.getNormal().normalize()
                    for l in lights:
                        lco = l.getIntensity(intersectionPoint,surfaceNormal)
                        h2 = r.Hit()

                        # Use 'try' to assume the light is directional,
                        # so shadows can be calculated

                        try:
                            #mode = 'directional'
                            ray2 = r.Ray(l.direction.normalize(),intersectionPoint)
                            h2 = group.intersect(ray2,0.001,dist_max)
                            if h2.t == float('inf'):
                                lsum.x += h.color.x * lco.x
                                lsum.y += h.color.y * lco.y
                                lsum.z += h.color.z * lco.z

                        # if code throws an attribute error, light is ambient
                        # (ambient lights have no attribute "direction")
                        except AttributeError:
                            #mode = 'ambient'
                            lsum.x += h.color.x * lco.x
                            lsum.y += h.color.y * lco.y
                            lsum.z += h.color.z * lco.z

                    # Check lsum values (cap at 1)
                    if lsum.x > 1:lsum.x = 1
                    if lsum.y > 1:lsum.y = 1
                    if lsum.z > 1:lsum.z = 1

                    color = lsum

                else:
                    print("invalid render mode")
                    pass
            else:
                color = bgc

            # color up until this point must be a Vec3 between 0-1
            color = color.scalarMult(255)
            color = (color.x, color.y, color.z)
            ra.setPixel(x,y,color)

    ra.display()

#-------------------------------------------------------
# Demos
#-------------------------------------------------------

def sphere_demo():
    persp = PerspectiveCamera(r.Vec3(0,0,0), r.Vec3(0,1,0), r.Vec3(0,0,-1), 32, 24, r.Vec3(16,12,50))
    s1 = Sphere(r.Vec3(16,12,10), 5, r.Vec3(1,0,0))
    s2 = Sphere(r.Vec3(13,13,15), 3, r.Vec3(0,0,1))
    s3 = Sphere(r.Vec3(14,9,12),4,r.Vec3(0,1,0))

    cTest = CSG(s1,s2,Operation.DIFFERENCE)
    cTest2 = CSG(cTest,s3,Operation.DIFFERENCE)

    group = Group()
    #group.addObject(s1)
    # group.addObject(s2)
    #group.addObject(cTest)
    group.addObject(cTest2)

    l1 = AmbientLight(r.Vec3(0.3, 0.3, 0.3))
    l2 = DirectionalLight(r.Vec3(1,-0.5,1),r.Vec3(0.2,0.7,0.2))
    l3 = DirectionalLight(r.Vec3(1,2,1),r.Vec3(0,0,0.3))
    l4 = DirectionalLight(r.Vec3(-1,0,1),r.Vec3(0.6,0.1,0.1))
    l5 = DirectionalLight(r.Vec3(1,2,1),r.Vec3(1,1,1))

    #lights = [l1,l2,l3]
    lights = [l1,l5]

    generateImage(320, 240, persp, group, 0, 55, lights, RenderMode.SHADOWS,r.Vec3(1,1,1))

def cube_demo():
    persp = PerspectiveCamera(r.Vec3(0,0,0), r.Vec3(0,1,0), r.Vec3(0,0,-1), 32, 24, r.Vec3(16,12,50))
    c = Cube(r.Vec3(7,8,10), 4, r.Vec3(1,0,0))
    c2 = Cube(r.Vec3(7,8,12),2,r.Vec3(0,1,0))
    t = Transformation(c2)
    t.setTransform(m.Matrix([[1,0,0,0],[0,1,0,0],[0,0,3,0],[0,0,0,1]]))
    s = Sphere(r.Vec3(7,8,10),2.8,r.Vec3(0,0,1))
    s2 = Sphere(r.Vec3(7,8,10),1,r.Vec3(0,1,0))
    csg = CSG(c,s,Operation.INTERSECTION)
    csg2 = CSG(csg,s2,Operation.DIFFERENCE)
    csg3 = CSG(csg2,t,Operation.DIFFERENCE)

    
    #t.setTransform(m.Matrix([[math.cos(-45),0,math.sin(-45),0],[0,1,0,0],[-(math.sin(-45)),0,math.cos(-45),0],[0,0,0,1]]))

    group = Group()
    group.addObject(csg3)

    l1 = AmbientLight(r.Vec3(0.3, 0.3, 0.3))
    l2 = DirectionalLight(r.Vec3(1,-0.5,1),r.Vec3(0.2,0.7,0.2))
    l3 = DirectionalLight(r.Vec3(1,2,1),r.Vec3(0,0,0.3))
    l4 = DirectionalLight(r.Vec3(-1,0,1),r.Vec3(0.6,0.1,0.1))
    l5 = DirectionalLight(r.Vec3(0,0,-1),r.Vec3(1,1,1))

    lights = [l1,l2,l3,l5]
    #lights = [l1,l5]

    generateImage(320, 240, persp, group, 0, 55, lights, RenderMode.DIFFUSE,r.Vec3(1,1,1))    

def final_demo():
    persp = PerspectiveCamera(r.Vec3(0,0,0), r.Vec3(0,1,0), r.Vec3(0,0,-1), 32, 24, r.Vec3(16,12,50))
    c = Cube(r.Vec3(7,14,10), 4, r.Vec3(1,0,0))
    c2 = Cube(r.Vec3(9.5,14,10),5,r.Vec3(1,0.5,0))
    
    s = Sphere(r.Vec3(7,14,10),2.8,r.Vec3(0,0,1))
    s2 = Sphere(r.Vec3(7,14,10),2.2,r.Vec3(0,1,0))

    csg = CSG(c,s,Operation.INTERSECTION)
    csg2 = CSG(csg,c2,Operation.DIFFERENCE)
    #csg3 = CSG(csg2, s2, Operation.DIFFERENCE)


    group = Group()
    group.addObject(csg2)

    l1 = AmbientLight(r.Vec3(0.3, 0.3, 0.3))
    l2 = DirectionalLight(r.Vec3(1,-0.5,1),r.Vec3(0.2,0.7,0.2))
    l3 = DirectionalLight(r.Vec3(1,2,1),r.Vec3(0,0,0.3))
    l4 = DirectionalLight(r.Vec3(-1,0,1),r.Vec3(0.6,0.1,0.1))
    l5 = DirectionalLight(r.Vec3(0,0,-1),r.Vec3(1,1,1))

    lights = [l1,l2,l3,l4,l5]
    #lights = [l1,l5]

    generateImage(320, 240, persp, group, 0.001, 55, lights, RenderMode.SHADOWS,r.Vec3(1,1,1))    




if __name__ == '__main__':sphere_demo()
