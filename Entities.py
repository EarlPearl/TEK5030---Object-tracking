import cv2
import numpy as np
class Entity:

    def __init__(self, pos, epoch, color=(0,0,255)):
        self.pos = [pos]
        self.last_updated = epoch
        self.queued_point = None
        self.color = color
        self.id = (str(np.random.choice([chr(i) for i in range(ord("A"), ord("Z"))], 2)).strip("[]").replace("'","") \
            + "-" + str(np.random.choice([i for i in range(10)], 5)).strip("[]").replace("'","")).replace(" ", "")

    def queue(self, point):
        if self.queued_point is None:
            self.queued_point = point
            return True
        return False

    def update(self, epoch):
        #TODO, check feature matches.
        #if none, check difference with last seen area
        if not self.queued_point is None:
            self.pos.append(self.queued_point)
            self.queued_point = None
            self.last_updated = epoch
        return self.last_updated
    
    def draw(self, frame):
        for point in self.pos:
            cv2.circle(frame, point, 1, self.color, 2)

    @property
    def x(self):
        return self.pos[-1][0]
    @property
    def y(self):
        return self.pos[-1][1]
        

class Entities:
    def __init__(self, MAX_ENTS=100, threshold = 5, decay=100):
        self.entities = []
        self.queue = []
        self.epoch = 0
        self.MAX_ENTS = MAX_ENTS
        self.threshold = threshold
        self.decay = decay
    
    def queue_point(self, point):
        #TODO maybe make it so that points are optimised not just given to the first entity within bounds?
        x, y = point
        for ent in self.entities:
            if (x < ent.x + self.threshold and \
                x > ent.x - self.threshold and \
                y < ent.y + self.threshold and \
                y > ent.y - self.threshold):
                if ent.queue(point):
                    return
        self.queue.append(point)
    
    def update(self):
        for ent in self.entities:
            if ent.update(self.epoch) <= self.epoch - self.decay:
                self.entities.remove(ent)

        while len(self.queue) > 0 and len(self.entities) < self.MAX_ENTS:
            point = self.queue.pop()
            self.entities.append(Entity(point, self.epoch))

        self.epoch += 1

    def draw(self, frame):
        for ent in self.entities:
            ent.draw(frame)