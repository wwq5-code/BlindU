

import numpy as np
import matplotlib.pyplot as plt
import math
epsilon = 3
beta = 1 / epsilon


# validation_for_plt =[97,95.8600, 94.9400, 93.5400, 93.2400]
# attack_for_plt=[0, 0.3524, 0, 0.1762, 0.1762]
# basic_for_plt=[99.8, 99.8, 99.8, 99.8, 99.8]

x=[0, 20, 40, 60, 80]
# validation_for_plt =[97,95.8600, 94.9400, 93.5400, 93.2400]
# attack_for_plt=[0, 0.3524, 0, 0.1762, 0.1762]
# basic_for_plt=[99.8, 99.8, 99.8, 99.8, 99.8]

#labels = ['0%','20%', '40%', '60%', '80%', '99%' ]
labels = ['0.0001', '0.001', '0.01', '0.1', '1' ]


unl_org_0= [0, 0, 0, 0, 0]
unl_org = [182.17, 199.2, 183.12, 128.49, 203.3]


unl_hess_r = [22.75, 22.75, 22.75, 22.75, 22.75]
unl_vbu = [199.53, 199.53, 199.53, 199.53, 199.53]

unl_ss_w = [40.874, 42.91, 63.794, 78.73, 82.50]
unl_ss_wo = [14.364, 16.911, 34.18, 55.13, 63.90]

with_replacement = []
without_replacement = []
for i in range(len(x)):
    with_replacement.append( 1- math.pow(99/100,x[i]))
    without_replacement.append(x[i]/100)

print(with_replacement)
print(without_replacement)

plt.figure(figsize=(6, 5))
l_w=5
m_s=12
marker_s = 3
markevery=1
#plt.figure(figsize=(8, 5.3))
#plt.plot(x, unl_fr, color='blue', marker='^', label='Retrain',linewidth=l_w, markersize=m_s)

plt.plot(x, unl_org_0, linestyle=':', color='r',  marker='^', fillstyle='none', markevery=markevery,
         label='No-Protection', linewidth=l_w,  markersize=m_s, markeredgewidth=marker_s)
# plt.plot(x, unl_vbu, color='r',  marker='p',  label='No-Protection (VBU)',linewidth=l_w, markersize=m_s)


#plt.plot(x, unl_hess_r, linestyle='-.', color='k',  marker='D', fillstyle='none', markevery=markevery, label='Gradient (HBFU)',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)
#plt.plot(x, unl_hess_r, color='dodgerblue',  marker='o', linestyle='-.', label='Gradient (HBFU)',linewidth=l_w, markersize=m_s)
#plt.plot(x, without_replacement, color='palegreen',  marker='1',  label='RFU-SS',linewidth=l_w, markersize=m_s)

plt.plot(x, unl_ss_w, linestyle='-', color='b', marker='o', fillstyle='none', markevery=markevery,
         label='PriMU$_{w}$', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

plt.plot(x, unl_ss_wo, linestyle='--', color='g',  marker='s', fillstyle='none', markevery=markevery,
         label='PriMU',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)



#plt.plot(x, unl_ss_w, color='g',  marker='*',  label='PriMU$_{w}$',linewidth=l_w, markersize=m_s)





#plt.plot(x, unl_vibu, color='silver',  marker='d',  label='VIBU',linewidth=4,  markersize=10)

# plt.plot(x, y_sa03, color='r',  marker='2',  label='AAAI21 A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_sa05, color='darkblue',  marker='4',  label='AAAI21 A_acc, pr=0.5',linewidth=3, markersize=8)
# plt.plot(x, y_ma03, color='darkviolet',  marker='3',  label='FedMC A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_ma05, color='cyan',  marker='p',  label='FedMC A_acc, pr=0.5',linewidth=3, markersize=8)


plt.grid()
leg = plt.legend(fancybox=True, shadow=True)
# plt.xlabel('Malicious Client Ratio (%)' ,fontsize=16)
plt.ylabel('Average MSE', fontsize=20)
my_y_ticks = np.arange(0, 100.01, 20)
plt.yticks(my_y_ticks,fontsize=20)
plt.xlabel('$\\beta$', fontsize=20)

plt.xticks(x, labels, fontsize=20)
# plt.title('CIFAR10 IID')

#plt.title('(a) Inference of different $\\beta$', fontsize=20)
plt.legend(loc='best',fontsize=20)
plt.tight_layout()
#plt.title("MNIST")
plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('ablation_MSE_beta_on_MNIST.pdf', format='pdf', dpi=200)
plt.show()