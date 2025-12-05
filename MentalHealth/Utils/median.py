import math

def linear_median(arr):
	arr = sorted(arr)
	if len(arr) % 2 != 0:
		return arr[(len(arr))/2]
	else:
		return float(arr[len(arr)/2] + arr[(len(arr ) - 1)/2])/2

def candMedian(dataPoints):
	#Calculate the first candidate median as the geometric mean 
	tempX = 0.0
	tempY = 0.0
	tempZ = 0.0
	
	for i in range(0,len(dataPoints)): 
		tempX += dataPoints[i][0] 
		tempY += dataPoints[i][1]
		tempZ += dataPoints[i][2]
	return [tempX/len(dataPoints), tempY/len(dataPoints), tempZ/len(dataPoints)]

def numersum(testMedian,dataPoint):
	# Provides the denominator of the weiszfeld algorithm depending on whether you are adjusting the candidate x or y 
	return 1/math.sqrt((testMedian[0]-dataPoint[0])**2 + (testMedian[1]-dataPoint[1])**2 + (testMedian[2]-dataPoint[2])**2)

def denomsum(testMedian, dataPoints):
	# Provides the denominator of the weiszfeld algorithm 
	temp = 0.0
	for i in range(0,len(dataPoints)):
		temp += 1/math.sqrt((testMedian[0] - dataPoints[i][0])**2 + (testMedian[1] - dataPoints[i][1])**2 + (testMedian[2] - dataPoints[i][2])**2) 
	return temp

def objfunc(testMedian, dataPoints):
	# This function calculates the sum of linear distances from the current candidate median to all points # in the data set, as such it is the objective function we are minimising.
	temp = 0.0
	for i in range(0,len(dataPoints)):
		temp += math.sqrt((testMedian[0]-dataPoints[i][0])**2 + (testMedian[1]-dataPoints[i][1])**2 + (testMedian[2]-dataPoints[i][2])**2) 
	return temp

def objfunc_distances(distances):
	return sum(distances)

def denomsum_distances(distances):
	temp = 0.0
	for distance in distances:
		temp += 1/distance
	return temp

def getdistances(testMedian, dataPoints):
	distances = []
	for i in range(0,len(dataPoints)):
		distances.append(math.sqrt((testMedian[0]-dataPoints[i][0])**2 + (testMedian[1]-dataPoints[i][1])**2 + (testMedian[2]-dataPoints[i][2])**2)) 
	return distances

#Calculates the median of an array of 3D points	
def getmedian(dataPoints):
	testMedian = candMedian(dataPoints)
	numIter = 50
	last_objfunc = 0.0
	for x in range(0,numIter):
		#print objfunc(testMedian,dataPoints)
		distances = getdistances(testMedian, dataPoints)
		current_objfunc = objfunc_distances(distances)
		if math.fabs(last_objfunc - current_objfunc) < 1:
			break
		last_objfunc =  current_objfunc
		denom = denomsum_distances(distances) 
		nextx = 0.0
		nexty = 0.0
		nextz = 0.0
		
		for y in range(0,len(dataPoints)):
			numer = 1/distances[y]#numersum(testMedian,dataPoints[y])
			nextx += (dataPoints[y][0] * numer)/denom 
			nexty += (dataPoints[y][1] * numer)/denom
			nextz += (dataPoints[y][2] * numer)/denom
		testMedian = [nextx,nexty,nextz]
	
	return testMedian

#Calculate median absolute deviation

def get_MAD_for_points_with_median(points, median):
	diffs = []
	for element in points:
		diffs.append(math.sqrt((element[0] - median[0])**2 + (element[1] - median[1])**2 + (element[2] - median[2])**2))
	mad_python = linear_median(diffs)
	return mad_python

def getMAD(points):
	if len(points) < 2:
		return -1
	median = getmedian(points)
	return get_MAD_for_points_with_median(points, median)

def getMAD_median(arr):
	mad_python = -1
	median_value = arr[0]
	if len(arr) > 1:
		median_value = getmedian(arr)
		diffs = []
		for element in arr:
			diffs.append(math.sqrt((element[0] - median_value[0])**2 + (element[1] - median_value[1])**2 + (element[2] - median_value[2])**2))
		mad_python = linear_median(diffs)
	return (mad_python, median_value)


def getMean(arr):
	sum_x = 0
	sum_y = 0
	sum_z = 0
	for element in arr:
		sum_x = sum_x + element[0]
		sum_y = sum_y + element[1]
		sum_z = sum_z + element[2]
	div = len(arr)
	return (float(sum_x)/div, float(sum_y)/div, float(sum_z)/div)

def getSTD(arr):
	mean = getMean(arr)
	diffs = []
	for element in arr:
		diffs.append((element[0] - mean[0])**2 + (element[1] - mean[1])**2 + (element[2] - mean[2])**2)
	return math.sqrt(float(sum(diffs))/len(diffs))


