import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
#创建画布
fig = plt.figure(figsize=(8, 8))
#使用axisartist.Subplot方法创建一个绘图区对象ax
ax = axisartist.Subplot(fig, 111)
#将绘图区对象添加到画布中
fig.add_axes(ax)
ax.axis[:].set_visible(False)
ax.axis["x"] = ax.new_floating_axis(0,0)
ax.axis["x"].set_axisline_style("->", size = 1.0)
ax.axis["y"] = ax.new_floating_axis(1,0)
ax.axis["y"].set_axisline_style("-|>", size = 1.0)
ax.axis["x"].set_axis_direction("top")
ax.axis["y"].set_axis_direction("right")

x_data = np.linspace(-3,5,100000)
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.plot(x_data,np.sqrt(x_data**3.0+x_data+1),color='b')
plt.plot(x_data,-1.0*np.sqrt(x_data**3.0+x_data+1),color='b')
plt.show()
print("hello world")