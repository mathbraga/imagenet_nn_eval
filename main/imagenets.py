# MobileNetV2 = model_1
# ResNet50 = model_2
# VGG16 = model_3

import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import os

img_src_path = '../src/imagenet2012_obj/'
labels_path = '../src/labels/imagenet_classes.txt'
ground_truth_path = '../src/groundtruth/ILSVRC2012_val.txt'
# imgname = '../src/imagenet2012_obj/ILSVRC2012_val_00000004.JPEG'
image_src = os.listdir(img_src_path)
data_set_size = 25

# Load MobileNetV2
model_1 = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
model_1.eval()

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

# Preprocess function
preprocess = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])

# Read the categories
with open(labels_path, "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Function that predicts images 1 by 1 for given model, returns values from 0 to 1
def model_prediction(model):
	top_1_rate = 0.0
	top_5_rate = 0.0

	for i in range(data_set_size):
		input_image = Image.open(img_src_path + image_src[i])
		converted_image = input_image.convert(mode='RGB')
		input_tensor = preprocess(converted_image)
		input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
		with torch.no_grad():
			output = model(input_batch)
		probabilities = torch.nn.functional.softmax(output[0], dim=0)

		# Evaluate top 1 and top 5
		_, top5_catid = torch.topk(probabilities, 5)
		print(categories[top5_catid[0]])
		if top5_catid[0] == ground_truth[image_src[i]]:
			top_1_rate += 1
		if (top5_catid[0] == ground_truth[image_src[i]] or
			top5_catid[1] == ground_truth[image_src[i]] or
			top5_catid[2] == ground_truth[image_src[i]] or
			top5_catid[3] == ground_truth[image_src[i]] or
			top5_catid[4] == ground_truth[image_src[i]]):
			top_5_rate += 1
	
	return top_1_rate/data_set_size, top_5_rate/data_set_size

# Predict MobileNetV2
model_1_top_1, model_1_top_5 = model_prediction(model_1)
model_1_top_1_hits = model_1_top_1*100
model_1_top_5_hits = model_1_top_5*100
model_1_top_1_err = 100 - model_1_top_1_hits
model_1_top_5_err = 100 - model_1_top_5_hits
print("MobileNetV2 top 1 error: {}".format(model_1_top_1_err))
print("MobileNetV2 top 5 error: {}".format(model_1_top_5_err))
