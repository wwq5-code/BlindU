# Importing the required library
import matplotlib.pyplot as plt
import numpy as np


# Sample data for the main categories and their sub-categories
categories = ['A', 'B', 'C', 'D']
sub_categories = ['Sub 1', 'Sub 2']
values_sub1 = [23, 45, 56, 78]
values_sub2 = [17, 33, 47, 64]

x = np.arange(len(categories))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()

# Creating the bars for sub-category 1
bars1 = ax.bar(x - width/2, values_sub1, width, label='Sub 1')

# Creating the bars for sub-category 2
bars2 = ax.bar(x + width/2, values_sub2, width, label='Sub 2')

# Adding value labels on top of each bar for sub-category 1
for bar in bars1:
    height = bar.get_height()
    ax.annotate('{}'.format(height),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# Adding value labels on top of each bar for sub-category 2
for bar in bars2:
    height = bar.get_height()
    ax.annotate('{}'.format(height),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# Labeling
ax.set_xlabel('Main Categories')
ax.set_ylabel('Values')
ax.set_title('Values by Category and Sub-Category')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

ax.yaxis.grid(True) # Only horizontal grid (y-axis)
ax.set_axisbelow(True) # Ensure grid is behind the bar plots


# Display the plot
plt.show()
