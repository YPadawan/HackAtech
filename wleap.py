import asyncio
from asyncio.tasks import sleep
import time
from websockets import connect
import json
import numpy as np
import mpl_toolkits.mplot3d as plt3d
import threading

# plotting test
import matplotlib.pyplot as plt
from matplotlib import animation
#import NeuroLeap as nl

# https://github.com/HyroVitalyProtago/LeapMotionRemoteUnity/blob/main/RemoteController.cs

# check runtime config for websocket over remote network
# https://developer-archive.leapmotion.com/documentation/cpp/devguide/Leap_Configuration.html

def normalize(vector): # todo check magnitude != 0
    magn = magnitude(vector)
    return np.true_divide(vector, magn) if magn != 0 else vector

def magnitude(vector):
    return np.linalg.norm(vector)

class Vector(np.ndarray):
    def __new__(self, a):
        return np.asarray(a).view(self)

    def __getattr__(self, name):
        if name == 'x': return self[0]
        elif name == 'y': return self[1]
        elif name == 'z': return self[2]

# class Matrix():
#     def __init__(self):
        
#     def to_array_3x3(self, output = None):
#         if output is None:
#             output = [0]*9
#         output[0], output[1], output[2] = self.x_basis.x, self.x_basis.y, self.x_basis.z
#         output[3], output[4], output[5] = self.y_basis.x, self.y_basis.y, self.y_basis.z
#         output[6], output[7], output[8] = self.z_basis.x, self.z_basis.y, self.z_basis.z
#         return output

class Bone():
    TYPE_METACARPAL = 0
    TYPE_PROXIMAL = 1
    TYPE_INTERMEDIATE = 2
    TYPE_DISTAL = 3

    def __init__(self, prev_joint, next_joint, center, direction, length, width, type, basis):
        self.prev_joint = prev_joint
        self.next_joint = next_joint
        self.center = center
        self.direction = direction
        self.length = length
        self.width = width
        self.type = type
        self.basis = basis
        self.is_valid = True # todo

class Finger():
    TYPE_THUMB = 0
    TYPE_INDEX = 1
    TYPE_MIDDLE = 2
    TYPE_RING = 3
    TYPE_PINKY = 4
    WIDTH = .01

    def __init__(self, frameId, handId, fingerId, timeVisible, tipPosition, direction, width, length, isExtended, type, metacarpal, proximal, intermediate, distal):
        self.frame_id = frameId
        self.hand_id = handId
        self.id = fingerId
        self.time_visible = timeVisible
        self.tip_position = tipPosition
        self.direction = direction
        self.width = width
        self.length = length
        self.is_extended = isExtended
        self.type = type
        self.bones = [metacarpal, proximal, intermediate, distal]

    # def frame(self):
    # def hand(self):

    def bone(self, boneIx):
        return self.bones[boneIx]

class InvalidHand():
    def __init__(self):
        self.is_valid = False

class Hand():
    def __init__(self, frameID, id, confidence, grabStrength, grabAngle, pinchStrength,
        pinchDistance, palmWidth, isLeft, timeVisible, arm, fingers, palmPosition,
        stabilizedPalmPosition, palmVelocity, palmNormal, direction, wristPosition):
        self.id = id
        self.fingers = fingers
        self.palm_position = palmPosition
        self.palm_velocity = palmVelocity
        self.palm_normal = palmNormal
        self.direction = direction
        # Hand.basis is computed from the palmNormal and direction vectors
        # Basis vectors fo the arm property specify the orientation of a arm:
        # - arm.basis[0] – the x-basis. Perpendicular to the longitudinal axis of the arm; exits laterally from the sides of the wrist.
        # - arm.basis[1] – the y-basis or up vector. Perpendicular to the longitudinal axis of the arm; exits the top and bottom of the arm. More positive in the upward direction.
        # - arm.basis[2] – the z-basis. Aligned with the longitudinal axis of the arm. More positive toward the elbow.
        #
        # The bases provided for the right arm use the right-hand rule; those for the left arm use the left-hand rule.
        # Thus, the positive direction of the x-basis is to the right for the right arm and to the left for the left arm.
        # You can change from right-hand to left-hand rule by multiplying the basis vectors by -1.
        self.basis = np.array([
            np.cross(palmNormal,direction) if isLeft else np.cross(direction, palmNormal), # TODO check
            palmNormal,
            direction
        ])
        self.is_valid = True
        self.grab_angle = grabAngle
        self.pinch_distance = pinchDistance
        self.grab_strength = grabStrength
        self.pinch_strength = pinchStrength
        self.palm_width = palmWidth
        self.stabilized_palm_position = stabilizedPalmPosition
        self.wrist_position = wristPosition
        self.time_visible = timeVisible
        self.is_left = isLeft
        self.is_right = not isLeft
        self.frame_id = frameID
        self.arm = arm
        self.__confidence = confidence
    
    def finger(self, id):
        return self.fingers[id]

    def confidence(self):
        return self.__confidence

class Arm():
    def __init__(self, elbow, wrist, center, direction, length, width, basis):
        self.elbow = elbow
        self.wrist = wrist
        self.center = center
        self.direction = direction
        self.length = length
        self.width = width
        self.basis = basis

class HandList():
    def __init__(self):
        self.__list = []
        
        # not working properly, only there for backward compatibility, but please do not use
        self.is_empty = True
        self.leftmost = InvalidHand()
        self.rightmost = InvalidHand()
        self.frontmost = InvalidHand()
    
    def __len__(self):
        return len(self.__list)

    def __getitem__(self, index):
        return self.__list[index]

    def __iter__(self):
      _pos = 0
      while _pos < len(self.__list):
        yield self.__list[_pos]
        _pos += 1

    def __update_internal(self):
        self.is_empty = len(self) == 0
        for hand in self.__list:
            if not self.leftmost.is_valid or self.leftmost.palm_position[0] > hand.palm_position[0]:
                self.leftmost = hand
            if not self.rightmost.is_valid or self.rightmost.palm_position[0] < hand.palm_position[0]:
                self.rightmost = hand
            if not self.frontmost.is_valid or self.frontmost.palm_position[2] > hand.palm_position[2]:
                self.frontmost = hand

    def append(self, el):
        ret = self.__list.append(el)
        self.__update_internal()
        return ret

class Frame():
    def __init__(self, data=None):
        self.is_valid = False
        self.hands = HandList()
        if not data is None:
            self.load(data)

    def __create_bone(self, prevJoint, nextJoint, type, basis):
        return Bone(
            prev_joint = prevJoint,
            next_joint = nextJoint,
            center = (prevJoint + nextJoint) / 2,
            direction = normalize(nextJoint - prevJoint),
            length = magnitude(nextJoint - prevJoint),
            width = Finger.WIDTH,
            type = type,
            basis = basis
        )

    def load(self, data):
        self.is_valid = True
        self.id = data['id']
        self.timestamp = data['timestamp']
        self.current_frames_per_second = data['currentFrameRate']

        fingers = []
        for finger_data in data['pointables']:
            metacarpalBasePosition = np.array(finger_data["carpPosition"])
            proximalPhalanxBasePosition = np.array(finger_data["mcpPosition"])
            intermediatePhalanxBasePosition = np.array(finger_data["pipPosition"])
            distalPhalanxBasePosition = np.array(finger_data["dipPosition"])
            distalPhalanxTipPosition = np.array(finger_data["btipPosition"])
            bases = np.array(finger_data["bases"])

            fingers.append(Finger(
                frameId = self.id,
                handId = finger_data["handId"],
                fingerId = finger_data["id"],
                timeVisible = finger_data["timeVisible"],
                tipPosition = np.array(finger_data["tipPosition"]),
                direction = np.array(finger_data["direction"]),
                width = finger_data["width"],
                length = finger_data["length"],
                isExtended = finger_data["extended"],
                type = finger_data["type"],
                metacarpal = self.__create_bone(metacarpalBasePosition, proximalPhalanxBasePosition, Bone.TYPE_METACARPAL, bases[0]),
                proximal = self.__create_bone(proximalPhalanxBasePosition, intermediatePhalanxBasePosition, Bone.TYPE_PROXIMAL, bases[1]),
                intermediate = self.__create_bone(intermediatePhalanxBasePosition, distalPhalanxBasePosition, Bone.TYPE_INTERMEDIATE, bases[2]),
                distal = self.__create_bone(distalPhalanxBasePosition, distalPhalanxTipPosition, Bone.TYPE_DISTAL, bases[3])
            ))

        self.hands = HandList() # reset hand list (TODO optimize)
        for hand_data in data['hands']:
            hand_id = hand_data["id"]
            elbowPosition = np.array(hand_data["elbow"])
            wristPosition = Vector(hand_data["wrist"])

            hand_fingers = []
            for finger in fingers:
                if finger.hand_id == hand_id:
                    hand_fingers.append(finger)
            hand_fingers.sort(key=lambda f: f.type)

            self.hands.append(Hand(
                frameID = self.id,
                id = hand_id,
                confidence = hand_data["confidence"],
                grabStrength = hand_data["grabStrength"],
                grabAngle = 0, #hand_data["grabAngle"]
                pinchStrength = hand_data["pinchStrength"],
                pinchDistance = 0, #hand_data["pinchDistance"]
                palmWidth = hand_data["palmWidth"],
                isLeft = hand_data["type"] == "left",
                timeVisible = hand_data["timeVisible"],
                arm = Arm(
                    elbow = elbowPosition,
                    wrist = wristPosition,
                    center = (elbowPosition + wristPosition) / 2,
                    direction = normalize(wristPosition - elbowPosition),
                    length = magnitude(wristPosition - elbowPosition),
                    width = hand_data["armWidth"],
                    basis = np.array(hand_data["armBasis"])
                ),
                fingers = hand_fingers, # need to order by type because leap motion expect it (as detailed in doc: https://developer-archive.leapmotion.com/documentation/v2/csharp/api/Leap.Hand.html#csharpclass_leap_1_1_hand_1a34356976500331d2a1998cb6ad857dae)
                palmPosition = Vector(hand_data["palmPosition"]),
                stabilizedPalmPosition = np.array(hand_data["palmPosition"]), # TODO
                palmVelocity = np.array(hand_data["palmVelocity"]),
                palmNormal = np.array(hand_data["palmNormal"]),
                direction = np.array(hand_data["direction"]),
                wristPosition = wristPosition
            ))

class Device(): # TODO
    def __init__(self, deviceHandle, internalHandle, horizontalViewAngle, verticalViewAngle, range, baseline, type, isStreaming, serialNumber):
        self.deviceHandle = deviceHandle
        self.internalHandle = internalHandle
        self.horizontalViewAngle = horizontalViewAngle
        self.verticalViewAngle = verticalViewAngle
        self.range = range
        self.baseline = baseline
        self.type = type
        self.isStreaming = isStreaming
        self.serialNumber = serialNumber

class Controller():
    POLICY_BACKGROUND_FRAMES = 'background'
    #POLICY_IMAGES = LeapPython.Controller_POLICY_IMAGES
    POLICY_OPTIMIZE_HMD = 'optimizeHMD'
    #POLICY_ALLOW_PAUSE_RESUME = LeapPython.Controller_POLICY_ALLOW_PAUSE_RESUME
    #POLICY_RAW_IMAGES = LeapPython.Controller_POLICY_RAW_IMAGES
    #POLICY_MAP_POINTS = LeapPython.Controller_POLICY_MAP_POINTS

    def __init__(self, threaded=False):
        #self.__frames = [] # 60 last frames
        self.__current_frame = Frame()
        self.__devices = []
        if not threaded:
            self.__task = asyncio.create_task(self.wleap("ws://localhost:6437/v6.json")) # json versions: https://developer-archive.leapmotion.com/documentation/objc/supplements/Leap_JSON.html
        else:
            self.sem = threading.Semaphore()

    async def wleap(self, uri):
        async with connect(uri) as websocket:
            self.__wsend = websocket.send
            while True:
                self.on_message(await websocket.recv())

    async def wait_for_connected(self, timeout):
        mustend = time.time() + timeout
        while time.time() < mustend:
            if self.is_service_connected():
                return True
            await sleep(.005)
        return False

    def __send(self, msg):
        if self.is_service_connected():
            self.__wsend(msg)

    def on_message(self, msg):
        root = json.loads(msg)
        if 'version' in root:
            print(root) # {"serviceVersion":"2.3.1+33747", "version":6}
        elif 'event' in root:
            self.__on_device_event(root['event'])
        elif 'id' in root: # frame
            self.__on_frame(root)
        else:
            print('unknown message: ',msg)

    def __on_device_event(self, evt):
        print(evt)
        if evt['type'] == 'deviceEvent':
            self.__devices.append(Device(
                deviceHandle = 0,
                internalHandle = 0,
                horizontalViewAngle = 0,
                verticalViewAngle = 0,
                range = 0,
                baseline = 0,
                type = evt["state"]["type"],# == "Peripheral"
                isStreaming = evt["state"]["streaming"] == "true",
                serialNumber = evt["state"]["id"]
            ))
    
    def __on_frame(self, frame_data):
        self.__current_frame.load(frame_data)

    def is_service_connected(self):
        return hasattr(self, '__wsend')

    def frame(self): # history
        if hasattr(self, 'sem'): self.sem.acquire()
        ret = self.__current_frame
        if hasattr(self, 'sem'): self.sem.release()
        return ret

    def __update_policy_flags(self, flag, enable):
        self.__send('{{"{}":{}}}'.format(flag, enable))

    def set_policy_flags(self, flag):
        print('set policy flag {}'.format(flag))
        self.__update_policy_flags(flag, True)

async def LeapController():
    controller = Controller()
    await controller.wait_for_connected(10)
    return controller

async def runLeapTask(controller, uri):
    async with connect(uri) as websocket:
        controller.__wsend = websocket.send
        await controller.__wsend('{"background":true}') # TODO set background call
        while True:
            msg = await websocket.recv()
            controller.sem.acquire()
            controller.on_message(msg)
            controller.sem.release()

def runLeapController(controller):
    asyncio.run(runLeapTask(controller, "ws://localhost:6437/v6.json"))

async def LeapControllerThreaded():
    controller = Controller(threaded=True)
    leapThread = threading.Thread(target=runLeapController, args=(controller,), daemon = True) # Daemonic threads can’t be joined. However, they are destroyed automatically when the main thread terminates.
    leapThread.start()
    await controller.wait_for_connected(10)
    return controller

async def main():
    controller = await LeapController()
    controller.set_policy_flags(Controller.POLICY_OPTIMIZE_HMD)

    mustend = time.time() + 2
    while time.time() < mustend:
        frame = controller.frame()
        print(frame.id, frame.current_frames_per_second, len(frame.hands))
        await sleep(1/120)

def get_points(controller):
	frame = controller.frame()
	hand = frame.hands.rightmost
	if not hand.is_valid: return None
	fingers = hand.fingers

	X = []
	Y = []
	Z = []

	# Add the position of the palms
	X.append(-1 *hand.palm_position.x)
	Y.append(hand.palm_position.y)
	Z.append(hand.palm_position.z)

	# Add wrist position
	X.append(-1 * hand.wrist_position.x)
	Y.append(hand.wrist_position.y)
	Z.append(hand.wrist_position.z)

	# Add Elbow
	#arm = hand.arm
	#X.append(arm.elbow_position.x)
	#Y.append(arm.elbow_position.y)
	#Z.append(arm.elbow_position.z)

	# Add fingers
	for finger in fingers:
		for b in range(0, 4):
			'''
			0 = JOINT_MCP – The metacarpophalangeal joint, or knuckle, of the finger.
			1 = JOINT_PIP – The proximal interphalangeal joint of the finger. This joint is the middle joint of a finger.
			2 = JOINT_DIP – The distal interphalangeal joint of the finger. This joint is closest to the tip.
			3 = JOINT_TIP – The tip of the finger.
			'''
			bone = finger.bone(b)
			X.append(-1 * bone.prev_joint[0])
			Y.append(bone.prev_joint[1])
			Z.append(bone.prev_joint[2])

	return np.array([X, Z, Y])

NUM_POINTS = 22
async def ideal_main_plot():
    controller = await LeapControllerThreaded()
    #controller.set_policy_flags(Controller.POLICY_OPTIMIZE_HMD)

    # while True:
    #     pass

    # Matplotlib Setup
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', xlim=(-300, 300), ylim=(-200, 400), zlim=(-300, 300))
    ax.view_init(elev=45., azim=122)

    points = np.zeros((3, NUM_POINTS))
    patches = ax.scatter(points[0], points[1], points[2], s=[20]*NUM_POINTS, alpha=1)

    def animate(i):
        # Reset the plot
        reset_plot(ax)

        points = get_points(controller)
        # print(points)

        if (points is not None):
            patches = ax.scatter(points[0], points[1], points[2], s=10, alpha=1)
            plot_points(points, patches)
            plot_bone_lines(points, ax)

    anim = animation.FuncAnimation(fig, animate, blit=False, interval=2)
    try:
        plt.show()
    except KeyboardInterrupt:
        sys.exit(0)

def reset_plot(ax):
	'''
	The Line plots will plot other eachother, as I make new lines instead of changing the data for the old ones
	TODO: Fix plot_simple and plot_lines so I don't need to do this.
	'''
	# Reset the plot
	ax.cla()
	# Really you can just update the lines to avoid this
	ax.set_xlim3d([-200, 200])
	ax.set_xlabel('X [mm]')
	ax.set_ylim3d([-200, 150])
	ax.set_ylabel('Y [mm]')
	ax.set_zlim3d([-100, 300])
	ax.set_zlabel('Z [mm]')

# Plotting the whole hand
def plot_bone_lines(points, ax):
	'''
	Plot the lines for the hand based on a full hand model.
	(22 points, 66 vars)
	'''
	mcps = []

	# Wrist
	wrist = points[:,1]

	# For Each of the 5 fingers
	for i in range(0,5):
		n = 4*i + 2

		# Get each of the bones
		mcp = points[:,n+0]
		pip = points[:,n+1]
		dip = points[:,n+2]
		tip = points[:,n+3]

		# Connect the lowest joint to the middle joint
		bot = plt3d.art3d.Line3D([mcp[0], pip[0]], [mcp[1], pip[1]], [mcp[2], pip[2]])
		ax.add_line(bot)

		# Connect the middle joint to the top joint
		mid = plt3d.art3d.Line3D([pip[0], dip[0]], [pip[1], dip[1]], [pip[2], dip[2]])
		ax.add_line(mid)

		# Connect the top joint to the tip of the finger
		top = plt3d.art3d.Line3D([dip[0], tip[0]], [dip[1], tip[1]], [dip[2], tip[2]])
		ax.add_line(top)

		# Connect each of the fingers together
		mcps.append(mcp)
	for mcp in range(0,4):
		line = plt3d.art3d.Line3D([mcps[mcp][0], mcps[mcp+1][0]],
								  [mcps[mcp][1], mcps[mcp+1][1]],
								  [mcps[mcp][2], mcps[mcp+1][2]])
		ax.add_line(line)
	# Create the right side of the hand joining the pinkie mcp to the "wrist"
	line = plt3d.art3d.Line3D([wrist[0], mcps[4][0]],
								  [wrist[1], mcps[3+1][1]],
								  [wrist[2], mcps[3+1][2]])
	ax.add_line(line)

	# Generate the "Wrist", note right side is not right.
	line = plt3d.art3d.Line3D([wrist[0], mcps[0][0]],
								  [wrist[1], mcps[0][1]],
								  [wrist[2], mcps[0][2]])
	ax.add_line(line)

	# Connext the left hand side of the index finger to the thumb.
	thumb_mcp = points[:,1+2]
	pinky_mcp = points[:,4+2]
	line = plt3d.art3d.Line3D([thumb_mcp[0], pinky_mcp[0]],
								  [thumb_mcp[1], pinky_mcp[1]],
								  [thumb_mcp[2], pinky_mcp[2]])
	ax.add_line(line)

def plot_points(points, scatter):
	scatter.set_offsets(points[:2].T)
	scatter.set_3d_properties(points[2], zdir='z')

async def main_plot():
    controller = await LeapController()
    #controller.set_policy_flags(Controller.POLICY_OPTIMIZE_HMD)

    # Matplotlib Setup
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', xlim=(-300, 300), ylim=(-200, 400), zlim=(-300, 300))
    ax.view_init(elev=45., azim=122)

    points = np.zeros((3, NUM_POINTS))
    patches = ax.scatter(points[0], points[1], points[2], s=[20]*NUM_POINTS, alpha=1)

    def animate(i):
        # Reset the plot
        reset_plot(ax)

        points = get_points(controller)

        if (points is not None):
            patches = ax.scatter(points[0], points[1], points[2], s=10, alpha=1)
            plot_points(points, patches)
            plot_bone_lines(points, ax)

        fig.canvas.draw()
        plt.pause(.001)

    #anim = animation.FuncAnimation(fig, animate, blit=False, interval=2)
    try:
        plt.show(block=False)

        mustend = time.time() + 10
        while time.time() < mustend:
            animate(0)
            await sleep(.001)
    except KeyboardInterrupt:
        sys.exit(0)


# -------- Main Program Loop -----------
if __name__ == '__main__':
    asyncio.run(ideal_main_plot())