# MobileNetV2 = model_1
# ResNet50 = model_2
# VGG16 = model_3

import tensorflow as tf
# import torch
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.preprocessing import image

# GPU config
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

img_src_path = '../src/imagenet2012_obj/'
labels_path = '../src/labels/imagenet_classes.txt'
ground_truth_path = '../src/groundtruth/ILSVRC2012_val.txt'
# imgname = 'ILSVRC2012_val_00000001.JPEG'
image_src = os.listdir(img_src_path)
data_set_size = 50000

model_1 = tf.keras.applications.MobileNetV2()
model_2 = tf.keras.applications.ResNet50()
model_3 = tf.keras.applications.VGG16()
# model_3.summary()

# Preprocess for models in format 224x224, axis 0
def prepare_img_model_224_224(img_src_path):
	image_set = []
	path = img_src_path
	for i in range(data_set_size):
		img = image.load_img(path + image_src[i], target_size=(224, 224))
		img_array = image.img_to_array(img)
		img_exp_dims = np.expand_dims(img_array, axis=0)
		image_set.append(img_exp_dims)
		print('...')
	
	images = tf.convert_to_tensor(image_set)
	images = tf.reshape(images, (data_set_size, 224, 224, 3))
	return images

# Preprocessed images 224x224, aixs 0
preproc_images_224_224 = prepare_img_model_224_224(img_src_path)

# Preprocess and predict for MobileNetV2
# preprocessed_img_model_1 = tf.keras.applications.mobilenet_v2.preprocess_input(preproc_images_224_224)
# prediction_model_1 = model_1.predict(preprocessed_img_model_1)
# results_model_1 = tf.keras.applications.mobilenet_v2.decode_predictions(prediction_model_1)
# # print(results_model_1)

# # Preprocess and predict for ResNet50
# preprocessed_img_model_2 = tf.keras.applications.resnet.preprocess_input(preproc_images_224_224)
# prediction_model_2 = model_2.predict(preprocessed_img_model_2)
# results_model_2 = tf.keras.applications.resnet.decode_predictions(prediction_model_2)
# # print(results_model_2)

# # Preprocess and predict for VGG16
# preprocessed_img_model_3 = tf.keras.applications.vgg16.preprocess_input(preproc_images_224_224)
# prediction_model_3 = model_3.predict(preprocessed_img_model_3)
# results_model_3 = tf.keras.applications.vgg16.decode_predictions(prediction_model_3)
# # print(results_model_3)

# # Mount ground truth file
# ground_truth = {}
# def mount_ground_truth(file):
# 	file_variable = open(file)
# 	all_lines_variable = file_variable.readlines()
# 	# print(type(all_lines_variable))
# 	# print(len(all_lines_variable))
# 	for line in all_lines_variable:
# 		list_line = line.split(' ')
# 		filename = list_line[0]
# 		label    = int(list_line[1])
# 		# print(filename, ' ' , label)
# 		ground_truth[filename] = label

# # Mount labels file
# labels = {}
# def mount_labels(file):
# 	i = 0
# 	file_variable = open(file)
# 	all_lines_variable = file_variable.readlines()
# 	for line in all_lines_variable:
# 		ln = line.replace('\n', '')
# 		labels[ln] = i
# 		i += 1

# mount_labels(labels_path)
# mount_ground_truth(ground_truth_path)

# # Function that returns top 1 precision
# def top_one_precision(result):
# 	correct = 0.0
# 	for i in range(len(result)):
# 		top_one_result = result[i][0][1].replace('_', ' ')
# 		if labels[top_one_result] == ground_truth[image_src[i]]:
# 			correct += 1

# 	precision = correct/data_set_size
# 	return precision

# # Function that returns top 5 precision
# def top_five_precision(result):
# 	correct = 0.0
# 	for i in range(len(result)):
# 		top_one_result = result[i][0][1].replace('_', ' ')
# 		top_two_result = result[i][1][1].replace('_', ' ')
# 		top_three_result = result[i][2][1].replace('_', ' ')
# 		top_four_result = result[i][3][1].replace('_', ' ')
# 		top_five_result = result[i][4][1].replace('_', ' ')
# 		if (labels[top_one_result] == ground_truth[image_src[i]] or 
# 		labels[top_two_result] == ground_truth[image_src[i]] or 
# 		labels[top_three_result] == ground_truth[image_src[i]] or
# 		labels[top_four_result] == ground_truth[image_src[i]] or
# 		labels[top_five_result] == ground_truth[image_src[i]]):
# 			correct += 1

# 	precision = correct/data_set_size
# 	return precision

# # Precisions MobileNetV2
# precision_model_1_top_1 = top_one_precision(results_model_1)
# precision_model_1_top_5 = top_five_precision(results_model_1)
# print('MobileNetV2 precision top 1: {}'.format(precision_model_1_top_1))
# print('MobileNetV2 precision top 5: {}'.format(precision_model_1_top_5))

# # Precisions ResNet50
# precision_model_2_top_1 = top_one_precision(results_model_2)
# precision_model_2_top_5 = top_five_precision(results_model_2)
# print('ResNet50 precision top 1: {}'.format(precision_model_2_top_1))
# print('ResNet50 precision top 5: {}'.format(precision_model_2_top_5))

# # Precisions VGG16
# precision_model_3_top_1 = top_one_precision(results_model_3)
# precision_model_3_top_5 = top_five_precision(results_model_3)
# print('VGG16 precision top 1: {}'.format(precision_model_3_top_1))
# print('VGG16 precision top 5: {}'.format(precision_model_3_top_5))