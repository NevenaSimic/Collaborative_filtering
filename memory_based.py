# author: Nevena Simic
from math import sqrt

def euclidean(rating1, rating2):
	distance = 0
	for key in rating1:
		if key in rating2:
			distance += pow(rating1[key] - rating2[key], 2)
	return sqrt(distance)