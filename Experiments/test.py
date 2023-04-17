

import numpy as np
import matplotlib.pyplot as plt

epsilon = 3
beta = 1 / epsilon





y_ss_back_acc  = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9899999499320984, 1.0, 1.0, 0.9899999499320984, 0.9899999499320984, 0.9799999594688416, 1.0, 0.9699999690055847, 0.9599999785423279, 0.9300000071525574, 0.949999988079071, 0.8700000047683716, 0.9300000071525574, 0.8199999928474426, 0.8100000023841858, 0.7999999523162842, 0.7599999904632568, 0.5999999642372131, 0.6200000047683716, 0.47999998927116394, 0.3499999940395355, 0.3999999761581421, 0.28999999165534973, 0.14999999105930328, 0.11999999731779099, 0.14000000059604645, 0.05999999865889549, 0.05999999865889549, 0.019999999552965164]


y_ss_acc       = [0.9799999594688416, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9899999499320984, 1.0, 1.0, 0.9899999499320984, 1.0, 0.9799999594688416, 0.9899999499320984, 1.0, 0.9899999499320984, 1.0, 0.9799999594688416, 1.0, 1.0, 0.9899999499320984, 1.0, 0.9599999785423279, 0.9799999594688416, 0.9899999499320984, 0.9699999690055847, 0.9699999690055847, 0.9599999785423279, 0.9399999976158142, 0.9699999690055847, 0.9699999690055847, 0.9599999785423279, 0.949999988079071, 0.9300000071525574, 0.9099999666213989, 0.9199999570846558, 0.8799999952316284, 0.8999999761581421, 0.9300000071525574, 0.85999995470047, 0.9399999976158142, 0.8999999761581421, 0.8799999952316284, 0.9300000071525574, 0.9099999666213989, 0.8999999761581421]




x=[]
y_org_s = []
y_retrain_s =[]
y_unl_ss_s =[]

y_ss_b = []
y_ss_a = []

t_i=1
for i in range(40):
    # print(np.random.laplace(0, 1)/10+0.2)
    x.append(i*t_i*1)
    #y_fkl[i] = y_fkl[i*2]*100
    y_ss_a.append(y_ss_acc[i*t_i]*100)
    y_ss_b.append(y_ss_back_acc[i*t_i]*100)



lw=3
plt.figure(figsize=(7, 5.2))
plt.plot(x, y_ss_a, color='orange', linestyle='--',   label='Comp. Unl.',linewidth=lw,  markersize=10)
plt.plot(x, y_ss_b, color='g', linestyle='-',  label='Comp. Unl.(bac.)',linewidth=lw, markersize=10)
# #plt.plot(x, y_fkl, color='g',  marker='+',  label='VRFL')
# plt.plot(x, y_retrain_s, color='r',  linestyle='-.',  label='Retraining',linewidth=lw, markersize=10)

# #plt.plot(x, y_vbu_b_acc_list, color='b', linestyle='--',   label='BFU (Er.)',linewidth=4,  markersize=10)
# plt.plot(x, y_vibu_ss_b_acc_list, color='y', linestyle='-',  label='RFU-SS (Er.)',linewidth=4, markersize=10)
# # #plt.plot(x, y_fkl, color='g',  marker='+',  label='VRFL')
# plt.plot(x, y_hbu_b_acc_list, color='grey',  linestyle='-.',  label='HBU (Er.)',linewidth=4, markersize=10)


#plt.plot(x, y_vbu_acc_list, color='orange', linestyle='--',  marker='x',  label='BFU',linewidth=4,  markersize=10)
#plt.plot(x, y_vibu_ss_acc_list, color='g',  marker='*',  label='BFU-SS',linewidth=4, markersize=10)
# #plt.plot(x, y_fkl, color='g',  marker='+',  label='VRFL')
#plt.plot(x, y_hbu_acc_list, color='r',  marker='p',  label='HBU',linewidth=4, markersize=10)

# plt.plot(x, unl_fr, color='blue', marker='^', label='Retrain',linewidth=4, markersize=10)
# plt.plot(x, unl_br, color='orange',  marker='x',  label='BFU',linewidth=4,  markersize=10)
# plt.plot(x, unl_self_r, color='g',  marker='*',  label='BFU-SS',linewidth=4, markersize=10)
# plt.plot(x, unl_hess_r, color='r',  marker='p',  label='HFU',linewidth=4, markersize=10)

# plt.plot(x, y_unl_s, color='b', marker='^', label='Normal Bayessian Fed Unlearning',linewidth=3, markersize=8)
# plt.plot(x, y_unl_self_s, color='r',  marker='x',  label='Self-sharing Fed Unlearning',linewidth=3, markersize=8)
# #plt.plot(x, y_fkl, color='g',  marker='+',  label='VRFL')
# plt.plot(x, y_hessian_30_s, color='y',  marker='*',  label='Unlearning INFOCOM22',linewidth=3, markersize=8)


# plt.grid()
leg = plt.legend(fancybox=True, shadow=True)
plt.xlabel('Epoch' ,fontsize=20)
plt.ylabel('Accuracy (%)' ,fontsize=20)
my_y_ticks = np.arange(0 ,101,20)
plt.yticks(my_y_ticks,fontsize=20)
my_x_ticks = np.arange(0, 50, 10)
plt.xticks(my_x_ticks,fontsize=20)
# plt.title('CIFAR10 IID')
plt.legend(loc='best',fontsize=16)

#plt.title('(a) $\it{EDR}$=6%, CIFAR10',fontsize=20)
plt.title('Compressive unlearning process on MNIST',fontsize=16)

plt.tight_layout()
#plt.title("Fashion MNIST")
plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('test_temp.png', dpi=200)
plt.show()