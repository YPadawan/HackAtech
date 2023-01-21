import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as plt3d

# from resources.Windows import Leap
from wleap import Finger

class Leap():
	class Matrix():
		def __init__(self) -> None:
			print('NO')
			pass

# Leap Motion Hand Animation
finger_bones = ['metacarpals', 'proximal', 'intermediate', 'distal']

def get_points(controller):
	'''
	Returns points for a simple hand model.
	18 point model. Finger tips + Palm
	'''
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

	for finger in fingers:
		# Add finger tip positions
		X.append(-1 * finger.tip_position.x)
		Y.append(finger.tip_position.y)
		Z.append(finger.tip_position.z)
	return np.array([X, Z, Y])

def get_basis_points(controller):
	'''
	Returns points for a simple hand model.
	Relative to the hand itself.
	18 point model. Finger tips + Palm
	This is the method reccomended from leap motion
	to get points from the hands point of refrence.
	It seems to loose a dimention.
	'''
	frame = controller.frame()
	hand = frame.hands.rightmost
	if not hand.is_valid: return None

	# Transforming finger coordinates into hands POV
	hand_x_basis = hand.basis.x_basis
	hand_y_basis = hand.basis.y_basis
	hand_z_basis = hand.basis.z_basis
	hand_origin = hand.palm_position
	hand_transform = Leap.Matrix(hand_x_basis, hand_y_basis, hand_z_basis, hand_origin)
	hand_transform = hand_transform.rigid_inverse()

	fingers = hand.fingers

	X = []
	Y = []
	Z = []

	# Add the position of the palms
	palm_pos = hand_transform.transform_point(hand.palm_position)
	X.append(-1 *palm_pos.x)
	Y.append(palm_pos.y)
	Z.append(palm_pos.z)

	for finger in fingers:
		transformed_position = hand_transform.transform_point(finger.tip_position)
		# Add finger tip positions
		X.append(-1 * transformed_position.x)
		Y.append(transformed_position.y)
		Z.append(transformed_position.z)
	return np.array([X, Z, Y])


def get_stable_points(controller):
	'''
	Returns points for a simple hand model.
	18 point model. Finger tips + Palm
	Uses stabalized positions, not reccomended for ML.
	'''
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

	for finger in fingers:
		# Add finger tip positions
		X.append(-1 * finger.stabilized_tip_position.x)
		Y.append(finger.stabilized_tip_position.y)
		Z.append(finger.stabilized_tip_position.z)
	return np.array([X, Z, Y])


def plot_points(points, scatter):
	scatter.set_offsets(points[:2].T)
	scatter.set_3d_properties(points[2], zdir='z')

def plot_simple(points, ax):
	'''
	Plot lines connecting the palms to the fingers, assuming thats the only data we get.
	Assumes we are using the 18 point tip model from get_points().
	'''
	# Get Palm Position
	palm = points[:,0]

	# For Each of the 5 fingers
	for n in range(1,6):
		# Draw a line from the palm to the finger tips
		tip = points[:,n]
		top = plt3d.art3d.Line3D([palm[0], tip[0]], [palm[1], tip[1]], [palm[2], tip[2]])
		ax.add_line(top)

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

def plot_finger_bones(points, ax):
	for i in range(0,5):
		for j in range(1,4):
			id = i*4+j+1
			ax.add_line(plt3d.art3d.Line3D([-points[0][id-1], points[0][id]], [points[2][id-1], points[2][id]], [points[1][id-1], points[1][id]]))

def get_bone_points(controller):
	'''
	Gets points for a full hand model. (22 points, 66 vars)
	Uses 4 joints for each finger and 3 for the thumb.
	Also uses Palm and Wrist position.
	Note this could be reduced to 21 points as the thumb has 1 less joint.
	'''
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
		for joint in range(0,4):
			'''
			0 = JOINT_MCP – The metacarpophalangeal joint, or knuckle, of the finger.
			1 = JOINT_PIP – The proximal interphalangeal joint of the finger. This joint is the middle joint of a finger.
			2 = JOINT_DIP – The distal interphalangeal joint of the finger. This joint is closest to the tip.
			3 = JOINT_TIP – The tip of the finger.
			'''
			X.append(-1 * finger.bone(joint).prev_joint[0])
			Y.append(finger.bone(joint).prev_joint[1])
			Z.append(finger.bone(joint).prev_joint[2])

	return np.array([X, Z, Y])

def get_rel_bone_points(controller):
	frame = controller.frame()
	hand = frame.hands.rightmost
	if not hand.is_valid:
		return None
	fingers = hand.fingers

	# Get hand transform
	hand_x_basis = hand.basis.x_basis
	hand_y_basis = hand.basis.y_basis
	hand_z_basis = hand.basis.z_basis
	hand_origin = hand.palm_position
	hand_transform = Leap.Matrix(hand_x_basis, hand_y_basis, hand_z_basis, hand_origin)
	hand_transform = hand_transform.rigid_inverse()

	X = []
	Y = []
	Z = []

	palm_pos = hand_transform.transform_point(hand.palm_position)
	X.append(palm_pos.x)
	Y.append(palm_pos.y)
	Z.append(palm_pos.z)

	# Add wrist position
	wrist_pos = hand_transform.transform_point(hand.wrist_position)
	X.append(wrist_pos.x)
	Y.append(wrist_pos.y)
	Z.append(wrist_pos.z)

	# Add fingers
	for finger in fingers:
		for joint in range(0,4):
			'''
			0 = JOINT_MCP – The metacarpophalangeal joint, or knuckle, of the finger.
			1 = JOINT_PIP – The proximal interphalangeal joint of the finger. This joint is the middle joint of a finger.
			2 = JOINT_DIP – The distal interphalangeal joint of the finger. This joint is closest to the tip.
			3 = JOINT_TIP – The tip of the finger.
			'''
			bone = finger.bone(joint)
			transformed_position = hand_transform.transform_point(bone.prev_joint)
			X.append(-1 * transformed_position[0])
			Y.append(transformed_position[1])
			Z.append(transformed_position[2])

	return np.array([X, Z, Y])

def get_basis_bone_points(controller):
	'''
	Gets points for a full hand model. (22 points, 66 vars)
	Relative to the hand itself.
	Uses 4 joints for each finger and 3 for the thumb.
	Also uses Palm and Wrist position.
	Note this could be reduced to 21 points as the thumb has 1 less joint.
	'''
	frame = controller.frame()
	hand = frame.hands.rightmost
	if not hand.is_valid: return None

	# Get hand transform
	hand_x_basis = hand.basis.x_basis
	hand_y_basis = hand.basis.y_basis
	hand_z_basis = hand.basis.z_basis
	hand_origin = hand.palm_position
	hand_transform = Leap.Matrix(hand_x_basis, hand_y_basis, hand_z_basis, hand_origin)
	hand_transform = hand_transform.rigid_inverse()

	fingers = hand.fingers

	X = []
	Y = []
	Z = []

	# Add the position of the palms
	# Transform palm position
	palm_pos = hand_transform.transform_point(hand.palm_position)
	X.append(palm_pos.x)
	Y.append(palm_pos.y)
	Z.append(palm_pos.z)

	# Add wrist position
	wrist_pos = hand_transform.transform_point(hand.wrist_position)
	X.append(wrist_pos.x)
	Y.append(wrist_pos.y)
	Z.append(wrist_pos.z)


	# Add fingers
	for finger in fingers:
		for joint in range(0,4):
			'''
			0 = JOINT_MCP – The metacarpophalangeal joint, or knuckle, of the finger.
			1 = JOINT_PIP – The proximal interphalangeal joint of the finger. This joint is the middle joint of a finger.
			2 = JOINT_DIP – The distal interphalangeal joint of the finger. This joint is closest to the tip.
			3 = JOINT_TIP – The tip of the finger.
			'''
			# Transform the finger
			transformed_position = hand_transform.transform_point(finger.joint_position(joint))
			X.append(transformed_position[0])
			Y.append(transformed_position[1])
			Z.append(transformed_position[2])

	return np.array([X, Z, Y])

def get_rotation_matrix(bone):
	basis = bone.basis
	# print(basis)
	x_basis = basis[0]#.x_basis
	y_basis = basis[1]#.y_basis
	z_basis = basis[2]#.z_basis
	# matrix = Leap.Matrix(x_basis, y_basis, z_basis).to_array_3x3()
	# matrix = np.reshape(matrix, newshape=(3, 3))
	# matrix = np.ndarray([x_basis, y_basis, z_basis], shape=(3,3))
	return np.matrix(bone.basis)#matrix

def get_angles_from_rot(rot_mat):
	"""
	Function from LearnOpenCV, Satya Mallick:
	https://www.learnopencv.com/rotation-matrix-to-euler-angles/
	https://github.com/spmallick/learnopencv/blob/master/RotationMatrixToEulerAngles/rotm2euler.py
	"""
	sy = math.sqrt(rot_mat[0, 0] * rot_mat[0, 0] + rot_mat[1, 0] * rot_mat[1, 0])
	singular = sy < 1e-6

	if not singular:
		x = math.atan2(rot_mat[2, 1], rot_mat[2, 2])
		y = math.atan2(-rot_mat[2, 0], sy)
		z = math.atan2(rot_mat[1, 0], rot_mat[0, 0])
	else:
		x = math.atan2(-rot_mat[1, 2], rot_mat[1, 1])
		y = math.atan2(-rot_mat[2, 0], sy)
		z = 0

	return [x,y,z]

def get_bone_angles(controller):
	frame = controller.frame()
	hand = frame.hands.rightmost
	if not hand.is_valid: return None
	fingers = hand.fingers

	angles = []
	for finger in fingers:
		for b in range(0,4):
			last_bone = finger.bone(b-1) if b != 0 else hand
			bone = finger.bone(b)
			# Generate rotation matrices from basis vectors
			last_bone_mat = get_rotation_matrix(last_bone)
			bone_mat = get_rotation_matrix(bone)
			# Get rotation matrix between bones, change of basis
			rot_mat = np.matmul(bone_mat, last_bone_mat.transpose())
			# Generate euler angles from rotation matrix
			angles.append(get_angles_from_rot(rot_mat))

	return angles

# return an array of 16 core angles
def get_bone_core_angles(controller):
	frame = controller.frame()
	hand = frame.hands.rightmost
	if not hand.is_valid: return None
	fingers = hand.fingers

	angles = []
	for finger in fingers:
		for b in range(0,4):
			last_bone = finger.bone(b-1) if b != 0 else hand
			bone = finger.bone(b)
			# Generate rotation matrices from basis vectors
			last_bone_mat = get_rotation_matrix(last_bone)
			bone_mat = get_rotation_matrix(bone)
			# Get rotation matrix between bones, change of basis
			rot_mat = np.matmul(bone_mat, last_bone_mat.transpose())
			# Generate euler angles from rotation matrix
			bone_angles = get_angles_from_rot(rot_mat)

			if finger.type == Finger.TYPE_THUMB:#Leap.Finger.TYPE_THUMB:
				if b == 1 or b == 2: # tmc or mcp
					angles.append(bone_angles[0]) # fe
					angles.append(bone_angles[1]) # aa
			else:
				if b == 1: # mcp
					angles.append(bone_angles[0]) # fe
					angles.append(bone_angles[1]) # aa
				elif b == 2: # pip
					angles.append(bone_angles[0])

	return np.array(angles)

# return an array of angles for the whole hand from 16 core angles
# fe: flexion-extension ; aa: adduction-abduction
# rules in degrees:
# 	0 <= pip <= 110
# 	-15 <= mcp.aa <= 15
# 	0 <= mcp.fe <= .5*pip
def get_all_bone_angles_from_core(core_angles):
	thumb_tmc_fe = core_angles[0]
	thumb_tmc_aa = core_angles[1]
	thumb_mcp_fe = core_angles[2]
	thumb_mcp_aa = core_angles[3]
	index_mcp_fe = core_angles[4]
	index_mcp_aa = core_angles[5]
	index_pip = core_angles[6]
	middle_mcp_fe = core_angles[7]
	middle_mcp_aa = core_angles[8]
	middle_pip = core_angles[9]
	ring_mcp_fe = core_angles[10]
	ring_mcp_aa = core_angles[11]
	ring_pip = core_angles[12]
	little_mcp_fe = core_angles[13]
	little_mcp_aa = core_angles[14]
	little_pip = core_angles[15]

	# return np.array([
	# 	[
	# 	[0, 0, 0],
	# 	[thumb_tmc_fe, thumb_tmc_aa, 0],
	# 	[thumb_mcp_fe, thumb_mcp_aa, 0],
	# 	[.5*thumb_mcp_fe, 0, 0],
	# 	],
	# 	[
	# 	[0, 0, 0],
	# 	[index_mcp_fe, index_mcp_aa, 0],
	# 	[index_pip, 0, 0],
	# 	[(2/3)*index_pip, 0, 0],
	# 	],
	# 	[
	# 	[0, 0, 0],
	# 	[middle_mcp_fe, middle_mcp_aa, 0],
	# 	[middle_pip, 0, 0],
	# 	[(2/3)*middle_pip, 0, 0],
	# 	],
	# 	[
	# 	[0, 0, 0],
	# 	[ring_mcp_fe, ring_mcp_aa, 0],
	# 	[ring_pip, 0, 0],
	# 	[(2/3)*ring_pip, 0, 0],
	# 	],
	# 	[
	# 	[0, 0, 0],
	# 	[little_mcp_fe, little_mcp_aa, 0],
	# 	[little_pip, 0, 0],
	# 	[(2/3)*little_pip, 0, 0]
	# 	]
	# ])
	return np.array([
		[[ 1.31046264e-08,  -4.51514921e-01, -1.10714839e+00],
		 [ thumb_tmc_fe,    thumb_tmc_aa,    4.43525931e-02],
		 [ thumb_mcp_fe,    thumb_mcp_aa,    -1.51374219e-05],
		 [ .5*thumb_mcp_fe, -3.48011090e-04, -4.61823540e-05]],

		[[ 1.51001853e-01,  -1.66816939e-01, -2.52599496e-02],
		 [ index_mcp_fe,    index_mcp_aa,    2.14017097e-03],
		 [ index_pip,       1.05834660e-02,  2.26880026e-03],
		 [ (2/3)*index_pip, 1.61607979e-02,  5.17828515e-03]],

		[[ 1.48955360e-01,   -2.95251073e-02,  1.46350746e-01],
		 [ middle_mcp_fe,    middle_mcp_aa,    -6.98252021e-03],
		 [ middle_pip,       5.21854329e-03,   9.45964320e-04],
		 [ (2/3)*middle_pip, 8.02984721e-03,   2.29385727e-03]],

		[[ 1.50006131e-01, 1.21617494e-01,  2.20302206e-01],
		 [ ring_mcp_fe,    ring_mcp_aa,     -8.31232798e-03],
		 [ ring_pip,       -9.78466469e-03, -1.87282535e-03],
		 [ (2/3)*ring_pip, -1.35027593e-02, -4.01373429e-03]],

		[[ 1.09821396e-01,   2.62327279e-01,  3.74564039e-01],
		 [ little_mcp_fe,    little_mcp_aa,   -2.36530601e-02],
		 [ little_pip,       -2.15288672e-02, -5.00986306e-03],
		 [ (2/3)*little_pip, -2.92980001e-02, -9.67397979e-03]]])

def get_rot_from_angles(theta) :
	# Calculates Rotation Matrix given euler angles.
	"""
	Function from LearnOpenCV, Satya Mallick:
	https://www.learnopencv.com/rotation-matrix-to-euler-angles/
	https://github.com/spmallick/learnopencv/blob/master/RotationMatrixToEulerAngles/rotm2euler.py
	"""
	x = np.array([
		[1,         0,                  0                   ],
		[0,         math.cos(theta[0]), -math.sin(theta[0]) ],
		[0,         math.sin(theta[0]), math.cos(theta[0])  ]
	])

	y = np.array([
		[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
		[0,                     1,      0                   ],
		[-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
	])

	z = np.array([
		[math.cos(theta[2]),    -math.sin(theta[2]),    0],
		[math.sin(theta[2]),    math.cos(theta[2]),     0],
		[0,                     0,                      1]
	])

	R = np.dot(z, np.dot(y, x))
	return R

# relative origin of fingers for right hand
fingers_rel_orig = [
	[-19.6307, -6.59409, 54.1453],
	[-12.1095, 13.6792, 45.7827],
	[-0.628944, 14.6466, 42.306],
	[10.7102, 11.9239, 41.1872],
	[21.4952, 5.17447, 44.1176]
]

# length of bones for each fingers
boneLength = [
	[0, 48.6369514465332, 33.17750549316406, 17.5335693359375],
	[72.72685241699219, 41.81882858276367, 23.510164260864258, 12.793498992919922],
	[68.96880340576172, 46.912025451660156, 27.665447235107422, 14.072121620178223],
	[61.922454833984375, 43.4868049621582, 26.948686599731445, 13.99288558959961],
	[57.320980072021484, 34.41537094116211, 19.01691436767578, 12.915475845336914]
]

def get_points_from_angles(angles):
	# palm and wrist 0 default
	X = [0,0]
	Y = [0,0]
	Z = [0,0]
	for finger in range(0,5):
		tmp = fingers_rel_orig[finger]#hand_transform.transform_point(hand.fingers[finger].bone(0).prev_joint)
		prev_pos = [-tmp[0],tmp[1],tmp[2]] #[0,0,0]
		prev_rot = get_rot_from_angles([0,math.pi,0 if finger != 0 else -math.pi*.6])#np.identity(3)
		for bone in range(0,4):
			# pitch = angles[finger,bone, 0]
			# yaw = angles[finger,bone, 1]
			# roll = angles[finger,bone, 2]
			theta = angles[finger,bone,:]#angles[finger,bone, :]

			rot_mat = prev_rot.dot(get_rot_from_angles(theta))

			bone_assume = np.array([0,0,boneLength[finger][bone]])
			new_bone = prev_pos + rot_mat.dot(bone_assume)

			x = -new_bone[0]
			y = new_bone[1]
			z = new_bone[2]

			X.append(x)
			Y.append(y)
			Z.append(z)

			prev_pos = new_bone
			prev_rot = rot_mat

	# Convert to a numpy array
	a_points = [X, Z, Y]
	return np.array(a_points)

def get_points_from_core_angles(core_angles):
	return get_points_from_angles(get_all_bone_angles_from_core(core_angles))

# Other Leap Funcs
def save_points(points,name='points.csv'):
	# Save one single row/frame to disk
	np.savetxt(name, points, delimiter=',')

# Myo Multiprocessing functions
def get_myodata_arr(myo, shared_arr):
	# ------------ Myo Setup ---------------
	def add_to_queue(emg, movement):
		for i in range(8):
			shared_arr[i] = emg[i]

	myo.add_emg_handler(add_to_queue)

	def print_battery(bat):
		print("Battery level:", bat)

	myo.add_battery_handler(print_battery)

	# Its go time
	myo.set_leds([128, 128, 0], [128, 128, 0])
	# Vibrate to know we connected okay
	myo.vibrate(1)

	# Wait to start
	# m.connect will wait until we get a connection, but the leap doesnt block
	try:
		while (True):
				myo.run()
	except KeyboardInterrupt:
		print("Quitting Myo worker")
		quit()

# Dataframe Creation

def data_to_simple_df(myo_data, leap_data):
	'''
	Takes myo and leap data, returns dataframe
	Assumes 18 point model. Finger tips + Palm
	'''
	myo_cols = ["Channel_1", "Channel_2", "Channel_3", "Channel_4", "Channel_5", "Channel_6", "Channel_7", "Channel_8"]
	leap_cols = []
	finger_names = ['Palm', 'Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
	# We can of course generate column names on the fly:
	# Note the ordering being different to the order we pack them in.
	for dim in ["x","y","z"]:
		for finger in finger_names:
			leap_cols.append(f"{finger}_tip_{dim}")

	myo_df = pd.DataFrame(myo_data, columns=myo_cols)
	leap_df = pd.DataFrame(leap_data, columns=leap_cols)

	df = myo_df.join(leap_df)
	return df

def data_to_bones_df(myo_data, leap_data):
	'''
	Takes myo and leap data, returns dataframe.
	Assumes full hand model. (22 points, 66 vars)
	Uses 4 joints for each finger and 3 for the thumb.
	Also uses Palm and Wrist position.
	Note this could be reduced to 21 points as the thumb has 1 less joint.
	'''
	myo_cols = ["Channel_1", "Channel_2", "Channel_3", "Channel_4", "Channel_5", "Channel_6", "Channel_7", "Channel_8"]
	leap_bone_columns = [
		"Palm_x", "Palm_y", "Palm_z",
		"Wrist_x", "Wrist_y", "Wrist_z",
		'Thumb_MCP_x', 'Thumb_MCP_y', 'Thumb_MCP_z',
		'Thumb_PIP_x', 'Thumb_PIP_y', 'Thumb_PIP_z',
		'Thumb_DIP_x', 'Thumb_DIP_y', 'Thumb_DIP_z',
		'Thumb_TIP_x', 'Thumb_TIP_y', 'Thumb_TIP_z',
		'Index_MCP_x', 'Index_MCP_y', 'Index_MCP_z',
		'Index_PIP_x', 'Index_PIP_y', 'Index_PIP_z',
		'Index_DIP_x', 'Index_DIP_y', 'Index_DIP_z',
		'Index_TIP_x', 'Index_TIP_y', 'Index_TIP_z',
		'Middle_MCP_x', 'Middle_MCP_y', 'Middle_MCP_z',
		'Middle_PIP_x', 'Middle_PIP_y', 'Middle_PIP_z',
		'Middle_DIP_x', 'Middle_DIP_y', 'Middle_DIP_z',
		'Middle_TIP_x', 'Middle_TIP_y', 'Middle_TIP_z',
		'Ring_MCP_x', 'Ring_MCP_y', 'Ring_MCP_z',
		'Ring_PIP_x', 'Ring_PIP_y', 'Ring_PIP_z',
		'Ring_DIP_x', 'Ring_DIP_y', 'Ring_DIP_z',
		'Ring_TIP_x', 'Ring_TIP_y', 'Ring_TIP_z',
		'Pinky_MCP_x', 'Pinky_MCP_y', 'Pinky_MCP_z',
		'Pinky_PIP_x', 'Pinky_PIP_y', 'Pinky_PIP_z',
		'Pinky_DIP_x', 'Pinky_DIP_y', 'Pinky_DIP_z',
		'Pinky_TIP_x', 'Pinky_TIP_y', 'Pinky_TIP_z'
		]

	myo_df = pd.DataFrame(myo_data, columns=myo_cols)
	leap_df = pd.DataFrame(leap_data, columns=leap_bone_columns)

	df = myo_df.join(leap_df)
	return df

def data_to_angles_df(myo_data, leap_data):
	'''
	Takes myo and leap data, returns dataframe.
	Assumes full hand model as 16 core angles
	'''
	myo_cols = ["Channel_1", "Channel_2", "Channel_3", "Channel_4", "Channel_5", "Channel_6", "Channel_7", "Channel_8"]
	leap_bone_columns = [
		'Thumb_TMC_fe', 'Thumb_tmc_aa', 'Thumb_mcp_fe', 'Thumb_mcp_aa',
		'Index_mcp_fe', 'Index_mcp_aa', 'Index_pip',
		'Middle_mcp_fe', 'Middle_mcp_aa', 'Middle_pip',
		'Ring_mcp_fe', 'Ring_mcp_aa', 'Ring_pip',
		'Little_mcp_fe', 'Little_mcp_aa', 'Little_pip'
		]

	myo_df = pd.DataFrame(myo_data, columns=myo_cols)
	leap_df = pd.DataFrame(leap_data, columns=leap_bone_columns)

	df = myo_df.join(leap_df)
	return df
