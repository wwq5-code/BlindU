

import numpy as np
import matplotlib.pyplot as plt

epsilon = 3
beta = 1 / epsilon


x=[1, 2, 3, 4, 5]
# validation_for_plt =[97,95.8600, 94.9400, 93.5400, 93.2400]
# attack_for_plt=[0, 0.3524, 0, 0.1762, 0.1762]
# basic_for_plt=[99.8, 99.8, 99.8, 99.8, 99.8]

labels = ['0.0001', '0.001', '0.01', '0.1', '1' ]
unl_org_0= [9.36, 20.76, 4.664, 0.397, 0.14]
unl_org = [117.658,  103.09, 86.82, 70.335, 90.96]


unl_hess_r = [4.951407, 4.9514073, 4.951407, 4.951407, 4.951407]
unl_vbu = [52.9582, 52.9582, 52.9582, 52.9582, 52.9582]

unl_ss_w = [141.664, 14.293, 10.5341, 1.16, 0.33]
unl_ss_wo = [159.9931, 23.47563, 11.4308, 1.927988, 1.3694]



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

# plt.plot(x, unl_vbu, linestyle=':', color='r',  marker='^', fillstyle='none', markevery=markevery,
#          label='VBU', linewidth=l_w,  markersize=m_s, markeredgewidth=marker_s)
#
# plt.plot(x, unl_hess_r, linestyle='-.', color='k',  marker='D', fillstyle='none', markevery=markevery,
#          label='HBFU',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)



#plt.plot(x, unl_vibu, color='silver',  marker='d',  label='VIBU',linewidth=4,  markersize=10)

# plt.plot(x, y_sa03, color='r',  marker='2',  label='AAAI21 A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_sa05, color='darkblue',  marker='4',  label='AAAI21 A_acc, pr=0.5',linewidth=3, markersize=8)
# plt.plot(x, y_ma03, color='darkviolet',  marker='3',  label='FedMC A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_ma05, color='cyan',  marker='p',  label='FedMC A_acc, pr=0.5',linewidth=3, markersize=8)


plt.grid()
leg = plt.legend(fancybox=True, shadow=True)
# plt.xlabel('Malicious Client Ratio (%)' ,fontsize=16)
plt.ylabel('MI of $I(Z_e;X_e)$' ,fontsize=24)
my_y_ticks = np.arange(0 ,201,40)
plt.yticks(my_y_ticks,fontsize=20)
plt.xlabel('$\\beta$' ,fontsize=20)

plt.xticks(x, labels, fontsize=20)
# plt.title('CIFAR10 IID')
plt.title('(j) Information in $Z_e$', fontsize=24)
plt.legend(loc='best',fontsize=20)
plt.tight_layout()
#plt.title("MNIST")
plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('celeba_mi_beta_curve.pdf', format='pdf', dpi=200)
plt.show()