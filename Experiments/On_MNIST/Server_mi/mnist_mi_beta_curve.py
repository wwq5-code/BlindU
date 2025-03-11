

import numpy as np
import matplotlib.pyplot as plt

epsilon = 3
beta = 1 / epsilon


x=[1, 2, 3, 4, 5]
# validation_for_plt =[97,95.8600, 94.9400, 93.5400, 93.2400]
# attack_for_plt=[0, 0.3524, 0, 0.1762, 0.1762]
# basic_for_plt=[99.8, 99.8, 99.8, 99.8, 99.8]

labels = ['0.0001', '0.001', '0.01', '0.1', '1' ]
unl_org_0= [62.36, 26.76, 17.664, 5.737, 3.64]
unl_org = [182.17, 199.2, 183.12, 128.49, 203.3]


unl_hess_r = [76.05, 76.05, 76.05, 76.05, 76.05]
unl_vbu = [199.53, 199.53, 199.53, 199.53, 199.53]

unl_ss_w = [35.51, 14.6, 6.65, 1.53, 0.658]
unl_ss_wo = [47.961, 31.33, 11.943, 2.402, 0.95]


plt.figure(figsize=(5.5, 5.3))
l_w=5
m_s=15
marker_s = 3
markevery=1
#plt.figure(figsize=(8, 5.3))
#plt.plot(x, unl_fr, color='blue', marker='^', label='Retrain',linewidth=l_w, markersize=m_s)
plt.plot(x, unl_ss_w, linestyle='-', color='b', marker='o', fillstyle='none', markevery=markevery,
         label='PriMU$_{w}$', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

#plt.plot(x, unl_ss_w, color='g',  marker='*',  label='PriMU$_{w}$',linewidth=l_w, markersize=m_s)
plt.plot(x, unl_ss_wo, linestyle='--', color='g',  marker='s', fillstyle='none', markevery=markevery,
         label='PriMU$_{w/o}$',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

#plt.plot(x, unl_vbu, linestyle=':', color='r',  marker='^', fillstyle='none', markevery=markevery, label='VBU', linewidth=l_w,  markersize=m_s, markeredgewidth=marker_s)

#plt.plot(x, unl_hess_r, linestyle='-.', color='k',  marker='D', fillstyle='none', markevery=markevery, label='HBFU',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

#plt.plot(x, unl_vibu, color='silver',  marker='d',  label='VIBU',linewidth=4,  markersize=10)

# plt.plot(x, y_sa03, color='r',  marker='2',  label='AAAI21 A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_sa05, color='darkblue',  marker='4',  label='AAAI21 A_acc, pr=0.5',linewidth=3, markersize=8)
# plt.plot(x, y_ma03, color='darkviolet',  marker='3',  label='FedMC A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_ma05, color='cyan',  marker='p',  label='FedMC A_acc, pr=0.5',linewidth=3, markersize=8)


plt.grid()
leg = plt.legend(fancybox=True, shadow=True)
# plt.xlabel('Malicious Client Ratio (%)' ,fontsize=16)
plt.ylabel('MI of $I(Z_e;X_e)$' ,fontsize=24)
my_y_ticks = np.arange(0 ,101,20)
plt.yticks(my_y_ticks,fontsize=20)
plt.xlabel('$\\beta$' ,fontsize=20)

plt.xticks(x, labels, fontsize=20)
# plt.title('CIFAR10 IID')
plt.title('(a) Information in $Z_e$', fontsize=24)

plt.legend(loc='best',fontsize=20)
plt.tight_layout()
#plt.title("MNIST")
plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('mnist_mi_beta_curve.pdf', format='pdf', dpi=200)
plt.show()