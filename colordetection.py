from sklearn.cluster import KMeans
from collections import Counter
from matplotlib import pyplot as plt 
import cv2

def rgbtohex(rgb):
	hex="#{:02x}{:02x}{:02x}".format(int(rgb[0]),int(rgb[1]),int(rgb[2]))
	return hex
	
def plot_image_info(path,k=6):
	img_bgr=cv2.imread(path)
	img_rgb=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
	resize_img_rgb=cv2.resize(img_rgb,(64,64),interpolation=cv2.INTER_AREA)
	img_list=resize_img_rgb.reshape((resize_img_rgb.shape[0]*resize_img_rgb.shape[1],3))
	clt=KMeans(n_clusters=k)
	labels=clt.fit_predict(img_list)
	labels_counts=Counter(labels)
	print(labels);
	total_count=sum(labels_counts.values());
	print(labels_counts)
	center_colors=list(clt.cluster_centers_)
	for i in labels_counts.keys():
		print(center_colors[i]/255)
		
	ordered_colors= [center_colors[i]/255 for i in labels_counts.keys()]
	color_labels= [rgbtohex(ordered_colors[i]*255) for i in labels_counts.keys()]
	plt.figure(figsize=(14,8))
	plt.subplot(221)
	plt.imshow(img_rgb)
	plt.axis('off')
	
	plt.subplot(222)
	plt.pie(labels_counts.values(),labels=color_labels,colors=ordered_colors,
	startangle=90)
	plt.axis('off')
	plt.show()
	
plot_image_info('test.jpg')