

import numpy as np
import matplotlib.pyplot as plt

epsilon = 3
beta = 1 / epsilon


y_bfu_acc = [2.90640926361084, 6.381667137145996, 5.361865997314453, 5.164310455322266, 5.402413845062256, 4.358552932739258, 3.7164900302886963, 4.008617401123047, 1.6251015663146973, 5.497019290924072, 6.826877593994141, 6.001331329345703, 1.6779484748840332, 9.360282897949219, 1.6711969375610352, 1.504644513130188, 4.616785049438477, 3.3323538303375244, 6.075416564941406, 3.8456497192382812, 4.972613334655762, 3.8190577030181885, 6.479625225067139, 0.9934285283088684, 2.6428723335266113, 5.468107223510742, 5.048480987548828, 6.256105422973633, 10.328584671020508, 3.1347475051879883, 3.209773302078247, 2.290097236633301, 6.673267364501953, 3.6208958625793457, 6.781280040740967, 12.784257888793945, 6.209792613983154, 8.138452529907227, 5.3038201332092285, 7.946474075317383, 7.343048095703125, 0.716110348701477, 9.716327667236328, 3.444540500640869, 7.417936325073242, 4.64296293258667, 4.202234268188477, 3.711545944213867, 9.17013168334961, 2.893789052963257, 4.113824844360352, 6.904426574707031, 6.462954998016357, 0.7431200742721558, 4.869304656982422, 7.161731719970703, 7.7333526611328125, 2.23081636428833, 19.033708572387695, 1.317046880722046]



primu_wo = [7.831614017486572, 9.697463035583496, 3.4689502716064453, 4.26216983795166, 5.243443012237549, 1.7646212577819824, 5.1940131187438965, 7.499311447143555, 6.293024063110352, 5.851590156555176, 9.423884391784668, 5.067818641662598, 0.708138108253479, 5.897953510284424, 2.251938581466675, 7.5957512855529785, 8.281579971313477, 5.477350234985352, 2.5159285068511963, 4.562263488769531, 6.008923530578613, 4.014269828796387, 5.431406021118164, 7.31325626373291, 0.6203938722610474, 10.03566837310791, 1.3274155855178833, 3.083190441131592, 3.8365564346313477, 10.43167781829834, 5.508098125457764, 4.495303630828857, 3.1763205528259277, 0.7002573609352112, 12.293340682983398, 5.9318342208862305, 1.528230905532837, 3.097682476043701, 2.587707996368408, 4.635287761688232, 9.403535842895508, 7.040051460266113, 6.601496696472168, 4.854031562805176, 4.39317512512207, 3.3511338233947754, 6.470029830932617, 0.7322947382926941, 1.2156591415405273, 7.5329084396362305, 3.8051247596740723, 1.7426517009735107, 7.444927215576172, 7.5699462890625, 8.13868522644043, 5.873655319213867, 0.9534187912940979, 9.464317321777344, 5.759515285491943, 11.455357551574707]




x=[]
y_unl_s = []
y_unl_self_s =[]
y_nips_rkl_s =[]

y_bfu_acc_list = []
y_primu_wo =[]
step_num = 10
for i in range(20):
    # print(np.random.laplace(0, 1)/10+0.2)
    x.append(i)
    #y_fkl[i] = y_fkl[i*2]*100
    acc_now = y_bfu_acc[i*1]
    acc_now_wo = primu_wo[i*1]
    for j in range(step_num):
        if y_bfu_acc[i*1+j+1] > 10:
            y_bfu_acc[i * 1 + j + 1] =  5

        if primu_wo[i*1+j+1] > 10:
            primu_wo[i * 1 + j + 1] = 5
        acc_now += y_bfu_acc[i*1+j+1]
        acc_now_wo += primu_wo[i*1+j+1]
    acc_avg = acc_now/step_num
    acc_avg_wo = acc_now_wo/step_num

    y_bfu_acc_list.append(acc_avg)
    y_primu_wo.append(acc_avg_wo)

    # y_hfu_back_acc[i] = y_hfu_back_acc[i]*100
    # y_hfu_acc[i] = y_hfu_acc[i]*100


plt.figure()
lw=5
# plt.plot(x, y_bfu_acc_list, color='orange',  marker='x',  label='BFU',linewidth=4,  markersize=10)
# plt.plot(x, y_primu_wo, color='b',  marker='x',  label='BFU2',linewidth=4,  markersize=10)
plt.plot(x, y_bfu_acc_list, color='g', linestyle='-',  label='PriMU$_{w}$',linewidth=lw, markersize=10)
# #plt.plot(x, y_fkl, color='g',  marker='+',  label='VRFL')
plt.plot(x, y_primu_wo, color='b', linestyle=(0,(3,1,1,1)),  label='PriMU$_{w/o}$',linewidth=lw, markersize=10)

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
plt.xlabel('Epochs' ,fontsize=20)
plt.ylabel('MI of $I(Z_e;Z_a)$' ,fontsize=20)
my_y_ticks = np.arange(0 ,11,2)
plt.yticks(my_y_ticks,fontsize=20)
my_x_ticks = np.arange(0, 21, 5)
plt.xticks(my_x_ticks,my_x_ticks*2,fontsize=20)
# plt.xticks(x,fontsize=20)
plt.title('(b) On CIFAR10',fontsize=20)
plt.legend(loc='best',fontsize=20)
plt.tight_layout()
#plt.title("Fashion MNIST")
plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('cifar_mutual_info_detail_acc.pdf', format='pdf', dpi=200)
plt.show()