import cv2
import numpy as np
from COLORS import COLORS

class Entity:
    """
    An entity (object) is manly just a list of positions.
        Properties:
            pos: returns a list of all positions stored.
            predicted_pos: returns the latest position.
            x: returns the latest x coordinate.
            y: returns the latest y coordinate.
        (note this is really the u,v coordinate, not the global X,Y)

        Functions:
            queue
            update
            draw
    """
    def __init__(self, pos, offset, epoch, MAX_POINTS, color=(0,0,255)):
        """
        Args:
            pos: initial position
            offset: width and hight to the bounding box of the object
            epoch: what frame the object was created on
            MAX_POINTS: how many coordinates the object should remember
            color: color of the object

        """
        self.age = 0 #number of times updated
        self.MAX_POINTS = MAX_POINTS
        self.offset = offset
        self._pos = []
        self.pos = pos
        self.last_updated = epoch
        self.queued_point = None
        self.color = color
        #id is just two random letters and five random numbers so the tracking looks cooler
        self.id = (str(np.random.choice([chr(i) for i in range(ord("A"), \
            ord("Z"))], 2)).strip("[]").replace("'","") \
            + "-" + \
            str(np.random.choice([i for i in range(10)], 5)).strip("[]").replace("'","")).replace(" ", "")

    def queue(self, point, offset):
        """
        Args:
            point: tuple or list of coordinates (x, y) of the center point of a contour.

            offset: the width and hight of the bounding box of the contour,
                    used to draw a box around the object.
        Returns:
            bool: True if point was successfully queued, False if another point was already queued
        """
        if self.queued_point is None:
            self.queued_point = (point, offset)
            return True
        return False

    def update(self, epoch):
        """
        Updates the positon and age of the entity, if a point is queued.
        Args:
            epoch: int how long (how many frames) has the program ran for.
        Returns:
            score: a numerical score used rank the objects.
        """
        if not self.queued_point is None:
            self.pos, self.offset = self.queued_point
            self.queued_point = None
            self.last_updated = epoch
            self.age += 1
    
    def draw(self, frame):
        """
        Args:
            frame: the frame the object should draw itself to.
        Returns:
            None
        """
        x, y = self.predicted_pos
        w, h = self.offset
        #draw bounding box

        cv2.rectangle(img=frame, pt1=(x-w, y-h), pt2=(x + w, y + h), color=(0, 255, 0), thickness=1)

        #draw path
        for point in self.pos:
            cv2.circle(frame, point, 1, self.color, 2)
        #draw current position
        cv2.circle(frame, (self.x, self.y), 1, self.color, 10)

        #draw lable
        text = f"{self.id}: {self.age}"
        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 1
        thickness = 1
        x = self.x - self.offset[0]
        y = self.y - self.offset[1]
        
        (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(frame, (x, y - 12), (x + w, y), self.color, -1)
        cv2.rectangle(frame, (x, y - 12), (x + w, y), (0,255,0), 1)
        cv2.putText(frame, text, (x, y), font, font_scale, (0,0,0), thickness)

    @property
    def pos(self):
        return self._pos
    @pos.setter
    def pos(self, new_pos):
        self._pos.append(new_pos)
        if len(self._pos) > self.MAX_POINTS:
            self._pos.pop(0)
    @property
    def predicted_pos(self):
        return self._pos[-1]
    @property
    def x(self):
        return self._pos[-1][0]
    @property
    def y(self):
        return self._pos[-1][1]
        

class Entities:
    def __init__(self, MAX_ENTS=10, threshold=50, decay=4, MAX_POINTS = 1000):
        self.entities = [] #list of tracked entites/objects
        self.queue = [] #list of points to be evalueted
        self.epoch = 0 #how many times the update functuin has been called
        self.MAX_ENTS = MAX_ENTS #number of entites to keep track of
        self.MAX_POINTS = MAX_POINTS #the number of positions an entity will store
        self.threshold = threshold**2 #distance threshold between points and objects
        self.decay = decay #how long should we remember things that dont move?
        self.colors = COLORS()

    def flush(self):
        """clears tracked entites and queued points"""
        self.entities = []
        self.queue = []

    def queue_point(self, point, offset):
        """Finds the closes object to a point and assigns the point to it
        if no object is close enough, create it creates a new one"""
        x1, y1 = point
        dists = {}
        for ent in self.entities:
            x2, y2 = ent.predicted_pos
            d = (x1 - x2)**2 + (y1 - y2)**2
            if d <= self.threshold:
                dists[d] = ent
  
        if len(dists.keys()) > 0:
            dists[min(dists.keys())].queue(point, offset)
        else:
            self.queue.append((point, offset))
        return

    def update(self):
        """
        updates all entities
        delete entites not updated in awhile
        create new entites of unassigned points
        """
        for ent in self.entities:
            ent.update(self.epoch)
            if self.epoch - (ent.last_updated + ent.age) > self.decay:
                self.entities.remove(ent)
                continue

        while len(self.queue) > 0 and len(self.entities) < self.MAX_ENTS:
            point, offset = self.queue.pop()
            self.entities.append(Entity(point, offset, self.epoch, self.MAX_POINTS, self.colors.next))

        self.epoch += 1

    def draw(self, frame):
        """draws all entites"""
        for ent in self.entities:
            ent.draw(frame)