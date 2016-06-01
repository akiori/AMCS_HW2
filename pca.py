import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from PIL import Image

# load data, find where is the digit of 3
myFile = open("G:\digit recognition\optdigits-orig.tra\optdigits-orig.tra")
data = myFile.readlines()
train = []
for i in range(1934):
	if int(data[53 + 33*i]) == 3:
		begin = 33 * i + 21
		line_list = []
		for j in range(32):
			for item in data[begin + j].strip():
				line_list.append(int(item))
		train.append(line_list)
train = np.array(train, dtype=np.int).T

# using svd to form pca
mean = np.mean(train, axis=1).reshape(train.shape[0], 1)
print(mean.shape)
print(train.shape)
mean_data = train - mean
print(mean_data.shape)
U, s, V = np.linalg.svd(mean_data, full_matrices=True)
result = np.dot(U[:,:2].T,mean_data)


xMin = min(result[0])
xMax = max(result[0])
yMin = min(result[1])
yMax = max(result[1])

x_list = np.linspace(xMin, xMax, 7)[1:6]
y_list = np.linspace(yMin, yMax, 7)[1:6]

location = []
for i in range(5):
	for j in range(5):
		anchorPoint = np.array([x_list[i],y_list[4-j]]) 
		distance = []
		for item in result.T:
			dis = np.linalg.norm(anchorPoint - item)
			distance.append(dis)
		loc = distance.index(min(distance))
		location.append(loc)

# plot
xScale   = MultipleLocator(2) 
xTxt = FormatStrFormatter('%5.1f') 
xMinorScale   = MultipleLocator(4) 

yScale   = MultipleLocator(2) 
yTxt = FormatStrFormatter('%1.1f') 
yMinorScale   = MultipleLocator(4) 

plt.scatter(result[0], result[1], 50, color ='green',marker = 'o')
for item in location:
	plt.scatter(result[0][item], result[1][item], 60, color ='red', marker = 's')

ax = subplot(111) 

ax.xaxis.set_major_locator(xScale)
ax.xaxis.set_major_formatter(xTxt)

ax.yaxis.set_major_locator(yScale)
ax.yaxis.set_major_formatter(yTxt)

ax.xaxis.set_minor_locator(xMinorScale)
ax.yaxis.set_minor_locator(yMinorScale)

ax.set_ylabel('second principal component')
ax.set_xlabel('first principal component')

ax.xaxis.grid(True, which='major') #x
ax.yaxis.grid(True, which='minor') 

plt.show()

#display all the number
data_r = 255*np.ones([2,172], dtype=np.uint8)
data_c = 255*np.ones([32,2], dtype=np.uint8)

data_matrix_r = data_r
for i in range(5):
	data_matrix_c = data_c
	for j in range(5):
		data_matrix_c = np.c_[data_matrix_c, 255*train[:,location[5*i+j]].reshape(32,32)]
		data_matrix_c = np.c_[data_matrix_c, data_c]
	data_matrix_r = np.r_[data_matrix_r, data_matrix_c]
	data_matrix_r = np.r_[data_matrix_r, data_r]

#a = (255-data_matrix_r)
a = data_matrix_r

cc = Image.new("RGB", (172, 172))
red = [0, 1, 34, 35, 68, 69, 102, 103, 136, 137, 170, 171]
#np.savetxt("C:\\Users\\LJH\\Desktop\\AMCS\\test.txt", a)
for i in range(172):
	for j in range(172):
		cc.putpixel([i, j], (255 - a[i,j], 255 - a[i,j], 255 - a[i,j]))

for i in red:
	for j in range(172):
		cc.putpixel([i, j], (255, 0, 0))
		cc.putpixel([j, i], (255, 0, 0))
		
cc.rotate(90).show()