import matplotlib.pyplot as plt
import numpy as np

# user num = 50
labels = ['20%', '40%', '60%', '80%', '100%']
unl_fr = [997 , 997, 997, 997, 997  ]

unl_hess_r = [26*997/20/50000*100 +997/20, 26*997/20/50000*100 +997/20, 26*997/20/50000*100 +997/20 , 26*997/20/50000*100 +997/20, 26*997/20/50000*100 +997/20 ]
unl_br = [26*997/20/50000*100,           26*997/20/50000*100  , 26*997/20/50000*100, 26*997/20/50000*100, 26*997/20/50000*100]

unl_vib = [25*997/20/50000*100*2,          50*997/20/50000*100*2     , 50*997/20/50000*100*2, 58*997/20/50000*100*2, 50*997/20/50000*100*2]
unl_self_r=[25*997/20/50000*100*2,       75*997/20/50000*100*2      , 52*997/20/50000*100*2, 21*997/20/50000*100*2, 17*997/20/50000*100*2]


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
plt.xlabel('$\it{SR}$', fontsize=20)
# ax.bar_label(rects1, padding=1)
# ax.bar_label(rects2, padding=3)
# ax.bar_label(rects3, padding=3)

plt.tight_layout()

plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('cifar100_rt_epsilon_bar.pdf', format='pdf', dpi=200)
plt.show()
