# Project A.R.G.U.S


## Files:
* **demo.py**
    - contains a demo of the program

* **Tracker.py**
    - Tracker: responsible for motion detection and updating the entities.

* **MotionDetectionScene.py**
    - Scene, detectes the aruco markers, estimates pose and calibrates the camera.

* **Pose.py**
    - ArucoPose, holds the pose and is used to convert between U,V and X,Y coordinates
    
* **Entites.py**
    - Entity: representation of an object in the scene.
    - Entities: manages the all instances of the Entiy class.

* **COLORS.py**
    - COLORS: list of xqcd colors and a inteface to use them.

* **utils.py**
    - ViewGui: window manager to display frames/images
    - drawGridImage: draws a grid on the mini-map
