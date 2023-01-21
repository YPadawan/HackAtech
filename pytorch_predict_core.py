import time
import multiprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as plt3d
# from keras.models import load_model
import joblib

import torch
from training import EncoderResNetDecoder

# from resources.Windows import Leap
# from wleap import LeapControllerThreaded
from pyomyo import Myo, emg_mode
import NeuroLeap as nl

channel_cols = ['Channel_1', 'Channel_2', 'Channel_3',
    'Channel_4', 'Channel_5', 'Channel_6', 'Channel_7', 'Channel_8']

# Myo Multiprocessing functions


def get_myodata_arr(shared_arr):
	myo = Myo(mode=emg_mode.RAW)  # PREPROCESSED)
	myo.connect()

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

# def lerp(a, b, t):
#   return (1 - t) * a + t * b

# -------- Main Program Loop -----------
if __name__ == '__main__':
	try:
		# Load the pytorch model
		model = EncoderResNetDecoder.load_from_checkpoint("./models/epoch=99-step=20700.ckpt")
		model.eval()

		# Start a Myo worker to put data into a shared array
		myo_arr = multiprocessing.Array('d', range(8))
		p = multiprocessing.Process(target=get_myodata_arr, args=(myo_arr,))
		p.start()

		# Matplotlib Setup
		print("Matplot Setup")
		fig = plt.figure(figsize=(20, 10), dpi=100)
		# The Prediction Leap Plot
		ax = fig.add_subplot(121, projection='3d', xlim=(-200, 300), ylim=(-200, 400), zlim=(-200, 200))
		ax.view_init(elev=45., azim=122)
		# The ground truth leap plot
		ax2 = fig.add_subplot(122, projection='3d', xlim=(-200, 300), ylim=(-200, 400), zlim=(-200, 200))
		ax2.view_init(elev=45., azim=122)

		# last_angles = None
		myo_datas = []
		def predict_hand_points(model):
			# Get the Myo data for model input
			myo_data = np.frombuffer(myo_arr.get_obj())/2500 # Normalisation
			
			myo_datas.append(myo_data)
			if (len(myo_datas) > 1000):
				myo_datas.pop(0)

			# Use input to generate predictions
			if (len(myo_datas) == 1000):
				pred = model(myo_datas)#
				# pred = predict_hand(myo_data, model, input_scaler, output_scaler)
				# print(pred)
				angles = nl.get_all_bone_angles_from_core(pred)
			else:
				angles = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
			
			
			# if (not last_angles is None):	
			# 	for a in angles:
			# 		print(a)
			# 		angles[a] = lerp(last_angles[a],angles[a],.5) # never right, but less jittle
			# last_angles = angles
				
			points = nl.get_points_from_angles(angles)

			return points

		def animate(i):
			# Prediction Plotting
			if (p.is_alive()):
				cpoints = predict_hand_points(model)
			else:
				print("Data gatherer has exited")
				quit()

			# Needed for plot_simple, for line plots
			nl.reset_plot(ax)

			if (cpoints is not None):
				patches = ax.scatter(cpoints[0], cpoints[1], cpoints[2], s=10, alpha=1)
				nl.plot_points(cpoints, patches)
				nl.plot_bone_lines(cpoints, ax)

			# Leap Motion Plotting
			# Note this should be done in another thread or weird things happen.
			# points = nl.get_rel_bone_points(controller)
			# nl.reset_plot(ax2)
			# if (points is not None):
			# 	truth = ax2.scatter(points[0], points[1], points[2], s=10, alpha=1)
			# 	nl.plot_points(points, truth)
			# 	nl.plot_bone_lines(points, ax2)

			ax.set_title('Prediction Plot', fontsize=22)
			# ax2.set_title('Ground Truth Plot', fontsize=22)

		anim = animation.FuncAnimation(fig, animate, blit=False, interval=2)
		plt.show()
	except KeyboardInterrupt:
		print("Quitting...")
		quit()