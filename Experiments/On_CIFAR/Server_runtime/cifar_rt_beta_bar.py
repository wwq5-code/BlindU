import matplotlib.pyplot as plt
import numpy as np

# user num = 50
labels = ['0.0001', '0.001', '0.01', '0.1', '1']
unl_fr = [10*20*2.76*2 , 10*20*2.76*2, 10*20*2.76*2 , 10*20*2.76*2, 10*20*2.76*2  ]

unl_hess_r = [31*100/5000*2.76*2 +27.6*2, 30*100/5000*2.76*2 +27.6*2, 32*100/5000*2.76 *2+27.6 *2, 36*100/5000*2.76 *2+27.6*2, 36*100/5000*2.76*2 +27.6 *2]
unl_br = [8*100/5000*2.76*2,           13*100/5000*2.76 *2     , 21*100/5000*2.76*2, 13*100/5000*2.76*2, 18*100/5000*2.76*2]

unl_vib = [15*100/5000*2.76*2*2,          20*100/5000*2.76*2 *2    , 15*100/5000*2.76*2*2, 30*100/5000*2.76*2*2, 21*100/5000*2.76*2*2]
unl_self_r=[18*100/5000*2.76*2*2,       25*100/5000*2.76*2  *2    , 18*100/5000*2.76*2*2, 24*100/5000*2.76*2*2, 22*100/5000*2.76*2*2]



x = np.arange(len(labels))  # the label locations
width = 0.6  # the width of the bars
# no_noise = np.around(no_noise,0)
# samping = np.around(samping,0)
# ldp = np.around(ldp,0)


plt.figure()
#plt.subplots(figsize=(8, 5.3))
#plt.bar(x - width / 2 - width / 8 + width / 8, unl_fr, width=0.168, label='Retrain', color='dodgerblue', hatch='/')
plt.bar(x - width / 2 - width / 8 + width / 8 , unl_br,   width=0.168, label='VBU', color='r', hatch='/')
plt.bar(x - width / 8 - width / 16, unl_vib, width=0.168, label='PriMU$_{w}$', color='cornflowerblue', hatch='*')
plt.bar(x + width / 8, unl_self_r, width=0.168, label='PriMU$_{w/o}$', color='g', hatch='x')
plt.bar(x + width / 2 - width / 8 + width / 16, unl_hess_r, width=0.168, label='HBFU', color='orange', hatch='\\')


# plt.bar(x - width / 2.5 ,  unl_br, width=width/3, label='VBU', color='orange', hatch='\\')
# plt.bar(x,unl_self_r, width=width/3, label='RFU-SS', color='g', hatch='x')
# plt.bar(x + width / 2.5,  unl_hess_r, width=width/3, label='HBU', color='tomato', hatch='-')


# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel('Running Time (s)', fontsize=20)
# ax.set_title('Performance of Different Users n')
plt.xticks(x, labels, fontsize=20)
# ax.set_xticklabels(labels,fontsize=15)

my_y_ticks = np.arange(0, 66.1, 10)
plt.yticks(my_y_ticks, fontsize=20)
# ax.set_yticklabels(my_y_ticks,fontsize=15)

plt.grid(axis='y')
plt.legend(loc='upper left', fontsize=20)
plt.xlabel('$\\beta$' ,fontsize=20)
# ax.bar_label(rects1, padding=1)
# ax.bar_label(rects2, padding=3)
# ax.bar_label(rects3, padding=3)
plt.title("(b) On CIFAR10",fontsize=24)
plt.tight_layout()

plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('cifar_rt_beta_bar.pdf', format='pdf', dpi=200)
plt.show()
