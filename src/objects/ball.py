import pybullet as p

def ball(position = [0.5, 0.5, 0.65], mass = 1, radius = 0.1):
    PB_BallRadius = radius
    PB_BallMass =  mass
    ballPosition = position
    ballOrientation = [0, 0, 0, 1]
    ballColShape = p.createCollisionShape(p.GEOM_SPHERE, radius=PB_BallRadius)
    ballVisualShapeId = p.createVisualShape(p.GEOM_SPHERE, radius=PB_BallRadius, rgbaColor=[1, 0.27, 0, 1])
    pb_ballId = p.createMultiBody(PB_BallMass, ballColShape, ballVisualShapeId, ballPosition, ballOrientation)
    return pb_ballId
