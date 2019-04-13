from numpy import *
import time
import matplotlib.pyplot as plt 
 
 
# 计算核值
def calcKernelValue(matrix_x, sample_x, kernelOption):
	kernelType = kernelOption[0]
	numSamples = matrix_x.shape[0]
	kernelValue = mat(zeros((numSamples, 1)))
	
	if kernelType == 'linear':
		kernelValue = matrix_x * sample_x.T
	elif kernelType == 'rbf':
		sigma = kernelOption[1]
		if sigma == 0:
			sigma = 1.0
		for i in range(numSamples):
			diff = matrix_x[i, :] - sample_x
			kernelValue[i] = exp(diff * diff.T / (-2.0 * sigma**2))
	else:
		raise NameError('Not support kernel type! You can use linear or rbf!')
	return kernelValue
 
 
# 计算给定的数据集和核类型的核矩阵
def calcKernelMatrix(train_x, kernelOption):
	numSamples = train_x.shape[0]
	kernelMatrix = mat(zeros((numSamples, numSamples)))
	for i in range(numSamples):
		kernelMatrix[:, i] = calcKernelValue(train_x, train_x[i, :], kernelOption)
	return kernelMatrix
 
 
# 来存储变量和数据
class SVMStruct:
	def __init__(self, dataSet, labels, C, toler, kernelOption):
		self.train_x = dataSet # 每一行代表一个样本
		self.train_y = labels  # 相应的标签
		self.C = C             # 松弛变量
		self.toler = toler     # 迭代终止条件
		self.numSamples = dataSet.shape[0] # 样本数
		self.alphas = mat(zeros((self.numSamples, 1))) # 所有样本的拉格朗日因子
		self.b = 0
		self.errorCache = mat(zeros((self.numSamples, 2)))
		self.kernelOpt = kernelOption
		self.kernelMat = calcKernelMatrix(self.train_x, self.kernelOpt)
 
		
# 计算alpha_k的误差
def calcError(svm, alpha_k):
	output_k = float(multiply(svm.alphas, svm.train_y).T * svm.kernelMat[:, alpha_k] + svm.b)
	error_k = output_k - float(svm.train_y[alpha_k])
	return error_k
 
 
# 优化alpha_k后更新alpha_k的误差
def updateError(svm, alpha_k):
	error = calcError(svm, alpha_k)
	svm.errorCache[alpha_k] = [1, error]
 
 
# 选择步骤最大的alpha_j
def selectAlpha_j(svm, alpha_i, error_i):
	svm.errorCache[alpha_i] = [1, error_i] # 标记为有效(已优化)
	candidateAlphaList = nonzero(svm.errorCache[:, 0].A)[0] # mat.A返回array
	maxStep = 0; alpha_j = 0; error_j = 0
 
	# 求迭代步长的最大值
	if len(candidateAlphaList) > 1:
		for alpha_k in candidateAlphaList:
			if alpha_k == alpha_i: 
				continue
			error_k = calcError(svm, alpha_k)
			if abs(error_k - error_i) > maxStep:
				maxStep = abs(error_k - error_i)
				alpha_j = alpha_k
				error_j = error_k
	# 如果第一次进入这个循环，我们随机选择alpha_j
	else:       	
		alpha_j = alpha_i
		while alpha_j == alpha_i:
			alpha_j = int(random.uniform(0, svm.numSamples))
		error_j = calcError(svm, alpha_j)
	
	return alpha_j, error_j
 
 
# 优化alpha_i和alpha_j的内部循环
def innerLoop(svm, alpha_i):
	error_i = calcError(svm, alpha_i)
 
	### 检查并找出违反KKT条件的阿尔法
	## 满足KKT条件
	# 1) yi*f(i) >= 1 and alpha == 0 (outside the boundary)
	# 2) yi*f(i) == 1 and 0<alpha< C (on the boundary)
	# 3) yi*f(i) <= 1 and alpha == C (between the boundary)
	## 违反KKT条件
	# 因为 y[i]*E_i = y[i]*f(i) - y[i]^2 = y[i]*f(i) - 1, 所以
	# 1) if y[i]*E_i < 0, so yi*f(i) < 1, if alpha < C, violate!(alpha = C will be correct) 
	# 2) if y[i]*E_i > 0, so yi*f(i) > 1, if alpha > 0, violate!(alpha = 0 will be correct)
	# 3) if y[i]*E_i = 0, so yi*f(i) = 1, it is on the boundary, needless optimized
	if (svm.train_y[alpha_i] * error_i < -svm.toler) and (svm.alphas[alpha_i] < svm.C) or\
		(svm.train_y[alpha_i] * error_i > svm.toler) and (svm.alphas[alpha_i] > 0):
 
		# step 1: 选择alpha_j
		alpha_j, error_j = selectAlpha_j(svm, alpha_i, error_i)
		alpha_i_old = svm.alphas[alpha_i].copy()
		alpha_j_old = svm.alphas[alpha_j].copy()
 
		# step 2: 计算alpha_j的L和H的边界
		if svm.train_y[alpha_i] != svm.train_y[alpha_j]:
			L = max(0, svm.alphas[alpha_j] - svm.alphas[alpha_i])
			H = min(svm.C, svm.C + svm.alphas[alpha_j] - svm.alphas[alpha_i])
		else:
			L = max(0, svm.alphas[alpha_j] + svm.alphas[alpha_i] - svm.C)
			H = min(svm.C, svm.alphas[alpha_j] + svm.alphas[alpha_i])
		if L == H:
			return 0
 
		# step 3: 计算eta(样本i和j相似度)
		eta = 2.0 * svm.kernelMat[alpha_i, alpha_j] - svm.kernelMat[alpha_i, alpha_i] \
				  - svm.kernelMat[alpha_j, alpha_j]
		if eta >= 0:
			return 0
 
		# step 4: 更新alpha_j
		svm.alphas[alpha_j] -= svm.train_y[alpha_j] * (error_i - error_j) / eta
 
		# step 5: 调整alpha_j
		if svm.alphas[alpha_j] > H:
			svm.alphas[alpha_j] = H
		if svm.alphas[alpha_j] < L:
			svm.alphas[alpha_j] = L
 
		# step 6: 如果alpha_j移动得不够，就返回		
		if abs(alpha_j_old - svm.alphas[alpha_j]) < 0.00001:
			updateError(svm, alpha_j)
			return 0
 
		# step 7: 优化aipha_j后更新alpha_i
		svm.alphas[alpha_i] += svm.train_y[alpha_i] * svm.train_y[alpha_j] \
								* (alpha_j_old - svm.alphas[alpha_j])
 
		# step 8: 更新阈值b
		b1 = svm.b - error_i - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
													* svm.kernelMat[alpha_i, alpha_i] \
							 - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
													* svm.kernelMat[alpha_i, alpha_j]
		b2 = svm.b - error_j - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
													* svm.kernelMat[alpha_i, alpha_j] \
							 - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
													* svm.kernelMat[alpha_j, alpha_j]
		if (0 < svm.alphas[alpha_i]) and (svm.alphas[alpha_i] < svm.C):
			svm.b = b1
		elif (0 < svm.alphas[alpha_j]) and (svm.alphas[alpha_j] < svm.C):
			svm.b = b2
		else:
			svm.b = (b1 + b2) / 2.0
 
		# step 9: 在优化alpha_i,alpha_j和b之后更新alpha_i,alpha_j的误差
		updateError(svm, alpha_j)
		updateError(svm, alpha_i)
 
		return 1
	else:
		return 0
 
 
# 主要训练过程
def trainSVM(train_x, train_y, C, toler, maxIter, kernelOption = ('rbf', 1.0)):
	# 计算训练时间
	startTime = time.time()
 
	# 初始化SVM的数据结构
	svm = SVMStruct(mat(train_x), mat(train_y), C, toler, kernelOption)
	
	# 开始训练
	entireSet = True
	alphaPairsChanged = 0
	iterCount = 0
	# 迭代终止条件:
	# 	Condition 1: 到达最大迭代
	# 	Condition 2: 遍历所有样本后没有alpha变化，即所有alpha(样本)都符合KKT条件
	while (iterCount < maxIter) and ((alphaPairsChanged > 0) or entireSet):
		alphaPairsChanged = 0
 
		# 更新所有的训练样本
		if entireSet:
			for i in range(svm.numSamples):
				alphaPairsChanged += innerLoop(svm, i)
			print ('---iter:%d entire set, alpha pairs changed:%d' % (iterCount, alphaPairsChanged))
			iterCount += 1
		# 更新alpha不为0和不为C(不在边界上)的样本的alpha
		else:
			nonBoundAlphasList = nonzero((svm.alphas.A > 0) * (svm.alphas.A < svm.C))[0]
			for i in nonBoundAlphasList:
				alphaPairsChanged += innerLoop(svm, i)
			print ('---iter:%d non boundary, alpha pairs changed:%d' % (iterCount, alphaPairsChanged))
			iterCount += 1
 
		# 交替循环遍历所有样本和非边界样本
		if entireSet:
			entireSet = False
		elif alphaPairsChanged == 0:
			entireSet = True
 
	print ('Congratulations, training complete! Took %fs!' % (time.time() - startTime))
	return svm
 
 
# 测试您的训练后的SVM模型给定的测试集
def testSVM(svm, test_x, test_y):
	test_x = mat(test_x)
	test_y = mat(test_y)
	numTestSamples = test_x.shape[0]
	supportVectorsIndex = nonzero(svm.alphas.A > 0)[0]
	supportVectors 		= svm.train_x[supportVectorsIndex]
	supportVectorLabels = svm.train_y[supportVectorsIndex]
	supportVectorAlphas = svm.alphas[supportVectorsIndex]
	matchCount = 0
	for i in range(numTestSamples):
		kernelValue = calcKernelValue(supportVectors, test_x[i, :], svm.kernelOpt)
		predict = kernelValue.T * multiply(supportVectorLabels, supportVectorAlphas) + svm.b
		if sign(predict) == sign(test_y[i]):
			matchCount += 1
	accuracy = float(matchCount) / numTestSamples
	return accuracy
 
 
# 训练支持向量机模型(只适用于二维数据)
def showSVM(svm):
	if svm.train_x.shape[1] != 2:
		print ("Sorry! I can not draw because the dimension of your data is not 2!")
		return 1
 
	# 画出所有样本
	for i in range(svm.numSamples):
		if svm.train_y[i] == -1:
			plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'or')
		elif svm.train_y[i] == 1:
			plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'ob')
 
	# 标记支持向量
	supportVectorsIndex = nonzero(svm.alphas.A > 0)[0]
	for i in supportVectorsIndex:
		plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'oy')
	
	# 画分界线
	w = zeros((2, 1))
	for i in supportVectorsIndex:
		w += multiply(svm.alphas[i] * svm.train_y[i], svm.train_x[i, :].T) 
	min_x = min(svm.train_x[:, 0])[0, 0]
	max_x = max(svm.train_x[:, 0])[0, 0]
	y_min_x = float(-svm.b - w[0] * min_x) / w[1]
	y_max_x = float(-svm.b - w[0] * max_x) / w[1]
	plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
	plt.show()



from numpy import *
import SVM

################## test svm #####################
## step 1: load data
print ("step 1: load data...")
dataSet = []
labels = []
fileIn = open('E:\\imok\\work\\CSP\\SVM\\testSet.txt')
for line in fileIn.readlines():
	lineArr = line.strip().split('\t')
	dataSet.append([float(lineArr[0]), float(lineArr[1])])
	labels.append(float(lineArr[2]))
 
dataSet = mat(dataSet)
labels = mat(labels).T
train_x = dataSet[0:81, :]
train_y = labels[0:81, :]
test_x = dataSet[80:101, :]
test_y = labels[80:101, :]
 
## step 2: training...
print ("step 2: training...")
C = 0.6
toler = 0.001
maxIter = 50
svmClassifier = SVM.trainSVM(train_x, train_y, C, toler, maxIter, kernelOption = ('linear', 0))
 
## step 3: testing
print ("step 3: testing...")
accuracy = SVM.testSVM(svmClassifier, test_x, test_y)
 
## step 4: show the result
print ("step 4: show the result..."	)
print ('The classify accuracy is: %.3f%%' % (accuracy * 100))
SVM.showSVM(svmClassifier)
