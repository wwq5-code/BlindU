import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
# Load the image

# image = np.transpose(image, (1, 2, 0))
# Display the image
# plt.imshow(image)
# plt.show()

x_coords=[1, 2, 3, 4, 5]

# validation_for_plt =[97,95.8600, 94.9400, 93.5400, 93.2400]
# attack_for_plt=[0, 0.3524, 0, 0.1762, 0.1762]
# basic_for_plt=[99.8, 99.8, 99.8, 99.8, 99.8]

labels = ['20%', '40%', '60%', '80%', '100%']
unl_org_0= [0, 0, 0, 0, 0]
unl_org = [182.17, 199.2, 183.12, 128.49, 203.3]


unl_hess_r = [44.45, 44.45, 44.45, 44.45, 44.45 ]
unl_vbu = [199.53, 199.53, 199.53, 199.53, 199.53]

unl_ss_w = [513.68, 460.564, 418.467, 377.468, 357.126]
unl_ss_wo = [508.860, 433.654, 368.199, 302.4007, 282.599]



fig, ax = plt.subplots(figsize=(7, 5))


# for x, y in zip(x_coords, unl_org_0):
#     ax.imshow(image, extent=[x-index_change/5*4, x+index_change/5*4, y, y+2*index_change/4*60], aspect='auto')
#
# for x, y in zip(x_coords, unl_ss_wo):
#     ax.imshow(image, extent=[x-index_change/5*4, x+index_change/5*4, y, y+2*index_change/4*60], aspect='auto')

index_change = 0.4

#load org
image = mpimg.imread('cifar_image_sr_0001_org_wo.png')
ax.imshow(image, extent=[x_coords[0]+0.05, x_coords[0]+2*index_change/5*4+0.05, unl_org_0[0], unl_org_0[0]+2*index_change/2*250], aspect='auto')
image = mpimg.imread('cifar_image_sr_001_org_wo.png')
ax.imshow(image, extent=[x_coords[1] +0.05, x_coords[1]+2*index_change/5*4+0.05, unl_org_0[1], unl_org_0[1]+2*index_change/2*250], aspect='auto')
image = mpimg.imread('cifar_image_sr_01_org_wo.png')
ax.imshow(image, extent=[x_coords[2] +0.05, x_coords[2]+2*index_change/5*4+0.05, unl_org_0[2], unl_org_0[2]+2*index_change/2*250], aspect='auto')
image = mpimg.imread('cifar_image_sr_1_org_wo.png')
ax.imshow(image, extent=[x_coords[3] +0.05, x_coords[3]+2*index_change/5*4+0.05, unl_org_0[3], unl_org_0[3]+2*index_change/2*250], aspect='auto')
image = mpimg.imread('cifar_image_sr_10_org_wo.png')
#ax.imshow(image, extent=[x_coords[4] +0.05, x_coords[4]+2*index_change/5*4+0.05, unl_org_0[4], unl_org_0[4]+2*index_change/4*60], aspect='auto')


#image = mpimg.imread('mnist_image_beta_0001_wo.png')
#ax.imshow(image, extent=[x_coords[0]-index_change/5*4, x_coords[0]+index_change/5*4, unl_ss_wo[0], unl_ss_wo[0]+2*index_change/4*60], aspect='auto')
image = mpimg.imread('cifar_image_sr_10_wn.png')
ax.imshow(image, extent=[x_coords[1]-2*index_change/5*4 -0.05, x_coords[1] -0.05, unl_ss_wo[1], unl_ss_wo[1]+2*index_change/2*250], aspect='auto')
image = mpimg.imread('cifar_image_sr_10_wo.png')
ax.imshow(image, extent=[x_coords[2]-2*index_change/5*4 -0.05, x_coords[2]-0.05 , unl_ss_wo[2], unl_ss_wo[2]+2*index_change/2*250], aspect='auto')
image = mpimg.imread('cifar_image_sr_1_wo.png')
ax.imshow(image, extent=[x_coords[3]-2*index_change/5*4-0.05, x_coords[3] -0.05, unl_ss_wo[3], unl_ss_wo[3]+2*index_change/2*250], aspect='auto')
image = mpimg.imread('cifar_image_sr_01_wo.png')
ax.imshow(image, extent=[x_coords[4]-2*index_change/5*4-0.05, x_coords[4]-0.05 , unl_ss_wo[4], unl_ss_wo[4]+2*index_change/2*250], aspect='auto')


l_w=5
m_s=10

# ax.scatter(x_coords, unl_org_0, marker='p', color='blue', zorder=20)
ax.scatter(x_coords, unl_ss_wo, marker='.', color='orange', zorder=2)
# ax.plot(x_coords, unl_ss_w, color='blue', zorder=3)
ax.plot(x_coords, unl_ss_w, color='limegreen',  marker='^', linestyle=(0,(2,1,1,1)), label='MCFU$_{w}$',linewidth=l_w, markersize=m_s)

ax.plot(x_coords, unl_ss_wo, color='orange',  marker='D', linestyle='--', label='MCFU$_{w/o}$',linewidth=l_w, markersize=m_s)

ax.plot(x_coords, unl_hess_r, color='dodgerblue',  marker='o', linestyle='-.', label='Grad. (HBFU)',linewidth=l_w, markersize=m_s)

ax.plot(x_coords, unl_org_0, color='r',  marker='p',  label='Masked Image',linewidth=l_w, markersize=m_s)

y_coords = [-10,10,20,30,666]
x1=[2,2,2,2,2]
ax.plot(x1, y_coords, color='gray',   linestyle='-',linewidth=1, markersize=m_s)
y_coords = [-10,10,20,30,666]
x1=[3,3,3,3,3]
ax.plot(x1, y_coords, color='gray',   linestyle='-',linewidth=1, markersize=m_s)
y_coords = [-10,10,20,30,666]
x1=[4,4,4,4,4]
ax.plot(x1, y_coords, color='gray',   linestyle='-',linewidth=1, markersize=m_s)



#ax.plot(x_coords, unl_vbu, color='orange',  marker='x',  label='VBU',linewidth=l_w,  markersize=m_s)

#ax.plot(x_coords, unl_hess_r, color='r',  marker='p',  label='HBU',linewidth=l_w, markersize=m_s)


# plt.title('CIFAR-10 image at specified points')
# 关闭自动扩展功能
plt.autoscale(False)
leg = plt.legend(fancybox=True, shadow=True)
# plt.xlabel('Malicious Client Ratio (%)' ,fontsize=16)
plt.ylabel('Average MSE' ,fontsize=20)
my_y_ticks = np.arange(0 ,560,100)
plt.yticks(my_y_ticks,fontsize=20)
ymin, ymax = plt.ylim()
plt.ylim(ymin - 0.05*(ymax-ymin), ymax + 0.05*(ymax-ymin))

plt.xlabel('$\it{SR}$' ,fontsize=20,fontstyle='italic')

plt.xticks(x_coords, labels, fontsize=20)
xmin, xmax = plt.xlim()
plt.xlim(xmin - 0.05*(xmax-xmin), xmax + 0.05*(xmax-xmin))

# plt.title('CIFAR10 IID') center right
plt.legend(loc='best',fontsize=20)
plt.tight_layout()
#plt.title("MNIST")
plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('cifar_sr_recontruction.png', dpi=200)
plt.show()



#plt.figure(figsize=(8, 5.3))
#plt.plot(x, unl_fr, color='blue', marker='^', label='Retrain',linewidth=l_w, markersize=m_s)


#plt.plot(x, unl_vibu, color='silver',  marker='d',  label='VIBU',linewidth=4,  markersize=10)

# plt.plot(x, y_sa03, color='r',  marker='2',  label='AAAI21 A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_sa05, color='darkblue',  marker='4',  label='AAAI21 A_acc, pr=0.5',linewidth=3, markersize=8)
# plt.plot(x, y_ma03, color='darkviolet',  marker='3',  label='FedMC A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_ma05, color='cyan',  marker='p',  label='FedMC A_acc, pr=0.5',linewidth=3, markersize=8)


# plt.grid()


# plt.show()