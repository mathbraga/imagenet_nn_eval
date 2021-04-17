# MobileNetV2 = model_1
# ResNet152 = model_2
# SqueezeNet1.0 = model_3
# VGG19 = model_4
# Alexnet = model_5
# GoogLeNet = model_6
# DenseNet201 = model_7
# InceptionV3 = model_8
# Shufflenet = model_9

import torch
# import tensorflow as tf
import numpy as np
import os
import sys
import time
# from tensorflow import keras
# from tensorflow.keras.preprocessing import image
from PIL import Image
from torchvision import transforms

img_src_path = '../src/imagenet2012_obj/' # ILSVRC2012 path
# img_src_path = '../src/extra_samples/' # personal samples path
labels_path = '../src/labels/imagenet_classes.txt'
ground_truth_path = '../src/groundtruth/ILSVRC2012_val.txt' # ILSVRC2012 ground truth path
# ground_truth_path = '../src/groundtruth/extra_samples_val.txt' # personal samples ground truth path
# imgname = '../src/imagenet2012_obj/ILSVRC2012_val_00000004.JPEG'
image_src = os.listdir(img_src_path)
image_src.sort()
data_set_size = 50000

# model_test = tf.keras.applications.MobileNetV2()
# model_test.summary()

# Mount ground truth file
ground_truth = {}
def mount_ground_truth(file):
	file_variable = open(file)
	all_lines_variable = file_variable.readlines()
	# print(type(all_lines_variable))
	# print(len(all_lines_variable))
	for line in all_lines_variable:
		list_line = line.split(' ')
		filename = list_line[0]
		label    = int(list_line[1])
		# print(filename, ' ' , label)
		ground_truth[filename] = label

# Mount labels file
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
mount_ground_truth(ground_truth_path)

# def prepare_img_model_1(img_src_path):
# 	path = img_src_path
# 	top_1_rate = 0.0
# 	top_5_rate = 0.0
# 	exec_time = 0.0
# 	for i in range(data_set_size):
# 		# Preprocess
# 		img = image.load_img(path + image_src[i], target_size=(224, 224))
# 		img_array = image.img_to_array(img)
# 		img_exp_dims = np.expand_dims(img_array, axis=0)
# 		target = tf.keras.applications.mobilenet_v2.preprocess_input(img_exp_dims)
# 		# Prediction
# 		start_time = time.time()
# 		prediction_model_test = model_test.predict(target)
# 		exec_time += time.time() - start_time
# 		# Results
# 		results_model_test = tf.keras.applications.mobilenet_v2.decode_predictions(prediction_model_test)
# 		top_one_result = results_model_test[0][0][1].replace('_', ' ')
# 		top_two_result = results_model_test[0][1][1].replace('_', ' ')
# 		top_three_result = results_model_test[0][2][1].replace('_', ' ')
# 		top_four_result = results_model_test[0][3][1].replace('_', ' ')
# 		top_five_result = results_model_test[0][4][1].replace('_', ' ')

# 		if labels[top_one_result] == ground_truth[image_src[i]]:
# 			top_1_rate += 1
# 		if (labels[top_one_result] == ground_truth[image_src[i]] or 
# 		labels[top_two_result] == ground_truth[image_src[i]] or 
# 		labels[top_three_result] == ground_truth[image_src[i]] or
# 		labels[top_four_result] == ground_truth[image_src[i]] or
# 		labels[top_five_result] == ground_truth[image_src[i]]):
# 			top_5_rate += 1
	
# 	precision_top_1 = top_1_rate/data_set_size
# 	precision_top_5 = top_5_rate/data_set_size
# 	return precision_top_1*100, precision_top_5*100, exec_time
		
# model_test_top_1, model_test_top_2, model_test_exec_time = prepare_img_model_1(img_src_path)
# print("Total time: {0:10.5f} s".format(model_test_exec_time))
# print("Average time per image: {0:10.5f} s".format(model_test_exec_time/data_set_size))
# print("MobileNetV2(tf) top 1 accuracy: {}".format(model_test_top_1))
# print("MobileNetV2(tf) top 5 accuracy: {}".format(model_test_top_2))

# Preprocess function
preprocess = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])

# Preprocess function exclusive for InceptionV3
preprocess_inception = transforms.Compose([
			transforms.Resize(299),
			transforms.CenterCrop(299),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])

# Read the categories
with open(labels_path, "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Function that predicts images 1 by 1 for given model, returns values from 0 to 1
def model_prediction(model, preprocess):
	top_1_rate = 0.0
	top_5_rate = 0.0
	exec_time = 0.0

	for i in range(data_set_size):
		# print(i)
		input_image = Image.open(img_src_path + image_src[i])
		converted_image = input_image.convert(mode='RGB')
		input_tensor = preprocess(converted_image)
		input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
		with torch.no_grad():
			start_time = time.time()
			output = model(input_batch)
			exec_time += time.time() - start_time
		probabilities = torch.nn.functional.softmax(output[0], dim=0)

		# Evaluate top 1 and top 5
		_, top5_catid = torch.topk(probabilities, 5)
		if top5_catid[0] == ground_truth[image_src[i]]:
			top_1_rate += 1
		if (top5_catid[0] == ground_truth[image_src[i]] or
			top5_catid[1] == ground_truth[image_src[i]] or
			top5_catid[2] == ground_truth[image_src[i]] or
			top5_catid[3] == ground_truth[image_src[i]] or
			top5_catid[4] == ground_truth[image_src[i]]):
			top_5_rate += 1
	
	print("Total time: {0:10.5f} s".format(exec_time))
	print("Average time per image: {0:10.5f} s".format(exec_time/data_set_size))
	return top_1_rate/data_set_size, top_5_rate/data_set_size


# Predict MobileNetV2
# if sys.argv[1] == 'model_1':
# Load MobileNetV2
model_1 = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
model_1.eval()

model_1_top_1, model_1_top_5 = model_prediction(model_1, preprocess)
model_1_top_1_acc = model_1_top_1*100
model_1_top_5_acc = model_1_top_5*100
model_1_top_1_err = 100 - model_1_top_1_acc
model_1_top_5_err = 100 - model_1_top_5_acc
print("MobileNetV2 top 1 accuracy: {}".format(model_1_top_1_acc))
print("MobileNetV2 top 5 accuracy: {}".format(model_1_top_5_acc))
print("MobileNetV2 top 1 error: {}".format(model_1_top_1_err))
print("MobileNetV2 top 5 error: {}".format(model_1_top_5_err))

# Predict ResNet50
# if sys.argv[1] == 'model_2':
# Load ResNet50
model_2 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet152', pretrained=True)
model_2.eval()

model_2_top_1, model_2_top_5 = model_prediction(model_2, preprocess)
model_2_top_1_acc = model_2_top_1*100
model_2_top_5_acc = model_2_top_5*100
model_2_top_1_err = 100 - model_2_top_1_acc
model_2_top_5_err = 100 - model_2_top_5_acc
print("ResNet152 top 1 accuracy: {}".format(model_2_top_1_acc))
print("ResNet152 top 5 accuracy: {}".format(model_2_top_5_acc))
print("ResNet152 top 1 error: {}".format(model_2_top_1_err))
print("ResNet152 top 5 error: {}".format(model_2_top_5_err))

# Predict SqueezeNet1.0
# if sys.argv[1] == 'model_3':
# Load SqueezeNet1.0
model_3 = torch.hub.load('pytorch/vision:v0.9.0', 'squeezenet1_0', pretrained=True)
model_3.eval()

model_3_top_1, model_3_top_5 = model_prediction(model_3, preprocess)
model_3_top_1_acc = model_3_top_1*100
model_3_top_5_acc = model_3_top_5*100
model_3_top_1_err = 100 - model_3_top_1_acc
model_3_top_5_err = 100 - model_3_top_5_acc
print("SqueezeNet1.0 top 1 accuracy: {}".format(model_3_top_1_acc))
print("SqueezeNet1.0 top 5 accuracy: {}".format(model_3_top_5_acc))
print("SqueezeNet1.0 top 1 error: {}".format(model_3_top_1_err))
print("SqueezeNet1.0 top 5 error: {}".format(model_3_top_5_err))

# Predict VGG16
# if sys.argv[1] == 'model_4':
# Load VGG16
model_4 = torch.hub.load('pytorch/vision:v0.9.0', 'vgg19', pretrained=True)
model_4.eval()

model_4_top_1, model_4_top_5 = model_prediction(model_4, preprocess)
model_4_top_1_acc = model_4_top_1*100
model_4_top_5_acc = model_4_top_5*100
model_4_top_1_err = 100 - model_4_top_1_acc
model_4_top_5_err = 100 - model_4_top_5_acc
print("VGG19 top 1 accuracy: {}".format(model_4_top_1_acc))
print("VGG19 top 5 accuracy: {}".format(model_4_top_5_acc))
print("VGG19 top 1 error: {}".format(model_4_top_1_err))
print("VGG19 top 5 error: {}".format(model_4_top_5_err))

# Predict Alexnet
# if sys.argv[1] == 'model_5':
# Load Alexnet
model_5 = torch.hub.load('pytorch/vision:v0.9.0', 'alexnet', pretrained=True)
model_5.eval()

model_5_top_1, model_5_top_5 = model_prediction(model_5, preprocess)
model_5_top_1_acc = model_5_top_1*100
model_5_top_5_acc = model_5_top_5*100
model_5_top_1_err = 100 - model_5_top_1_acc
model_5_top_5_err = 100 - model_5_top_5_acc
print("Alexnet top 1 accuracy: {}".format(model_5_top_1_acc))
print("Alexnet top 5 accuracy: {}".format(model_5_top_5_acc))
print("Alexnet top 1 error: {}".format(model_5_top_1_err))
print("Alexnet top 5 error: {}".format(model_5_top_5_err))

# Predict GoogLeNet
# if sys.argv[1] == 'model_6':
# Load GoogLeNet
model_6 = torch.hub.load('pytorch/vision:v0.9.0', 'googlenet', pretrained=True)
model_6.eval()

model_6_top_1, model_6_top_5 = model_prediction(model_6, preprocess)
model_6_top_1_acc = model_6_top_1*100
model_6_top_5_acc = model_6_top_5*100
model_6_top_1_err = 100 - model_6_top_1_acc
model_6_top_5_err = 100 - model_6_top_5_acc
print("GoogLeNet top 1 accuracy: {}".format(model_6_top_1_acc))
print("GoogLeNet top 5 accuracy: {}".format(model_6_top_5_acc))
print("GoogLeNet top 1 error: {}".format(model_6_top_1_err))
print("GoogLeNet top 5 error: {}".format(model_6_top_5_err))

# Predict DenseNet121
# if sys.argv[1] == 'model_7':
# Load DenseNet121
model_7 = torch.hub.load('pytorch/vision:v0.9.0', 'densenet201', pretrained=True)
model_7.eval()

model_7_top_1, model_7_top_5 = model_prediction(model_7, preprocess)
model_7_top_1_acc = model_7_top_1*100
model_7_top_5_acc = model_7_top_5*100
model_7_top_1_err = 100 - model_7_top_1_acc
model_7_top_5_err = 100 - model_7_top_5_acc
print("DenseNet201 top 1 accuracy: {}".format(model_7_top_1_acc))
print("DenseNet201 top 5 accuracy: {}".format(model_7_top_5_acc))
print("DenseNet201 top 1 error: {}".format(model_7_top_1_err))
print("DenseNet201 top 5 error: {}".format(model_7_top_5_err))

# Predict InceptionV3
# if sys.argv[1] == 'model_8':
# Load InceptionV3
model_8 = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True)
model_8.eval()

model_8_top_1, model_8_top_5 = model_prediction(model_8, preprocess_inception)
model_8_top_1_acc = model_8_top_1*100
model_8_top_5_acc = model_8_top_5*100
model_8_top_1_err = 100 - model_8_top_1_acc
model_8_top_5_err = 100 - model_8_top_5_acc
print("InceptionV3 top 1 accuracy: {}".format(model_8_top_1_acc))
print("InceptionV3 top 5 accuracy: {}".format(model_8_top_5_acc))
print("InceptionV3 top 1 error: {}".format(model_8_top_1_err))
print("InceptionV3 top 5 error: {}".format(model_8_top_5_err))

# Predict Shufflenet
# if sys.argv[1] == 'model_9':
# Load Shufflenet
model_9 = torch.hub.load('pytorch/vision:v0.9.0', 'shufflenet_v2_x1_0', pretrained=True)
model_9.eval()

model_9_top_1, model_9_top_5 = model_prediction(model_9, preprocess)
model_9_top_1_acc = model_9_top_1*100
model_9_top_5_acc = model_9_top_5*100
model_9_top_1_err = 100 - model_9_top_1_acc
model_9_top_5_err = 100 - model_9_top_5_acc
print("Shufflenet top 1 accuracy: {}".format(model_9_top_1_acc))
print("Shufflenet top 5 accuracy: {}".format(model_9_top_5_acc))
print("Shufflenet top 1 error: {}".format(model_9_top_1_err))
print("Shufflenet top 5 error: {}".format(model_9_top_5_err))