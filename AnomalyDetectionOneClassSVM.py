from sklearn import svm
from numpy import genfromtxt
import matplotlib.pyplot as plt

def read_dataset(filePath,delimiter=','):
    return genfromtxt(filePath, delimiter=delimiter)
# use the same dataset
tr_data = read_dataset('tr_server_data.csv')
clf = svm.OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
clf.fit(tr_data)
pred = clf.predict(tr_data)

# inliers are labeled 1, outliers are labeled -1
normal = tr_data[pred == 1]
abnormal = tr_data[pred == -1]

plt.figure()
plt.plot(normal[:,0],normal[:,1],'bx')
plt.plot(abnormal[:,0],abnormal[:,1],'ro')
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.show()
