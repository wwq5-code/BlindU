import matplotlib.pyplot as plt
import numpy as np

# user num = 50
labels = ['2%', '4%', '6%', '8%', '10%']
unl_fr = [10*10*0.22 *5, 10*10*0.22*5, 10*10*0.22*5 , 10*10*0.22*5 , 10*10*0.22 *5  ]

unl_hess_r = [138*20/6000*0.22*5 +2.2 *5, 283*20/6000*0.22*5 +2.2*5 , 338*20/6000*0.22*5 +2.2 *5, 389*20/6000*0.22*5 +2.2 *5, 439*20/6000*0.22*5 +2.2 *5]
unl_br = [34*20/6000*0.22 *5          , 64*20/6000*0.22   *5    , 43*20/6000*0.22  *5     , 79*20/6000*0.22*5, 89*20/6000*0.22*5]


unl_vib = [240*20/6000*0.22  *2 *5      , 240*20/6000*0.22 *2 *5    , 360*20/6000*0.22 *2 *5   , 360*20/6000*0.22*2*5, 400*20/6000*0.22*2*5]
unl_self_r = [180*20/6000*0.22   *2 *5  , 240*20/6000*0.22 *2 *5    , 360*20/6000*0.22 *2 *5    , 360*20/6000*0.22*2*5, 400*20/6000*0.22*2*5]


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

plt.grid(axis='y')
my_y_ticks = np.arange(0, 13.1, 2)
plt.yticks(my_y_ticks, fontsize=20)
# ax.set_yticklabels(my_y_ticks,fontsize=15)

plt.legend(loc='upper left', fontsize=20)
plt.xlabel('$\it{EDR}$' ,fontsize=20)
# ax.bar_label(rects1, padding=1)
# ax.bar_label(rects2, padding=3)
# ax.bar_label(rects3, padding=3)
plt.title("(a) On MNIST",fontsize=24)
plt.tight_layout()


plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('mnist_rt_er_bar.pdf', format='pdf', dpi=200)
plt.show()
