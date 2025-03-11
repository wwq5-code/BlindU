

import numpy as np
import matplotlib.pyplot as plt

epsilon = 3
beta = 1 / epsilon


x=[1, 2, 3, 4, 5]
# validation_for_plt =[97,95.8600, 94.9400, 93.5400, 93.2400]
# attack_for_plt=[0, 0.3524, 0, 0.1762, 0.1762]
# basic_for_plt=[99.8, 99.8, 99.8, 99.8, 99.8]

labels = ['0.2%', '0.4%', '0.6%', '0.8%', '1%' ]
unl_org = [51.47, 51.95, 51.85, 52.64, 50.55]

unl_hess_r = [91.36, 89.62, 87.56, 52.49, 38.64]
unl_vbu = [96.71, 94.53, 94.865, 82.37,48.42 ]

unl_ss_w = [97.34, 97.36, 97.36, 97.31, 96.93]
unl_ss_wo = [97.33, 97.24, 97.13, 97.02, 96.75]



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

plt.plot(x, unl_vbu, linestyle=':', color='r',  marker='^', fillstyle='none', markevery=markevery,
         label='VBU', linewidth=l_w,  markersize=m_s, markeredgewidth=marker_s)

plt.plot(x, unl_hess_r, linestyle='-.', color='k',  marker='D', fillstyle='none', markevery=markevery,
         label='HBFU',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)



# plt.plot(x, y_sa03, color='r',  marker='2',  label='AAAI21 A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_sa05, color='darkblue',  marker='4',  label='AAAI21 A_acc, pr=0.5',linewidth=3, markersize=8)
# plt.plot(x, y_ma03, color='darkviolet',  marker='3',  label='FedMC A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_ma05, color='cyan',  marker='p',  label='FedMC A_acc, pr=0.5',linewidth=3, markersize=8)


plt.grid()
leg = plt.legend(fancybox=True, shadow=True)
# plt.xlabel('Malicious Client Ratio (%)' ,fontsize=16)
plt.ylabel('Accuracy (%)' ,fontsize=24)
my_y_ticks = np.arange(0, 101, 20)
plt.yticks(my_y_ticks,fontsize=20)
plt.xlabel('$\it{EDR}$' ,fontsize=20)

plt.xticks(x, labels, fontsize=20)
# plt.title('CIFAR10 IID')

plt.title('(l) Utility Preservation', fontsize=24)
plt.legend(loc='best',fontsize=20)
plt.tight_layout()
#plt.title("MNIST")
plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('celeba_acc_er_curve.pdf', format='pdf', dpi=200)
plt.show()