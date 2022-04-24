import matplotlib.pyplot as plt
import numpy as np
# new_ticks = np.linspace(10,100,9)
# print(new_ticks)
# plt.xticks(new_ticks)
#x=(10,20,30,40,50,60,70,80,90,100)
x=np.linspace(0.2,1,5)
#任务丢弃率
y=[0,0,0.283387,0.6322,0.91]
y2=(0, 0.000977,0.234219,0.6506,0.9)
y3=(0,0,0,0.064,0.293)
y4=(0,0,0,0.051,0.203)
y5=(0,0,0,0.024,0.093)

plt.xlim(0.2,1)
# plt.ylim(0,1)
plt.plot(x,y,color="blue",marker="o",label="local")
plt.plot(x,y2,color="red",marker="x",label="offload")
plt.plot(x,y3,color="black",marker="D",label="DQN")
plt.plot(x,y4,color="gray",marker="H",label="DDPG")
plt.plot(x,y5,color="green",marker="*",label="PER-DDPG")
plt.xlabel("task arrival probability",fontproperties="SimSun")
plt.ylabel("radio of dropped tasks",fontproperties="SimSun")

plt.legend()
plt.savefig("radio of dropped tasks.png",dpi=500,bbox_inches="tight")
plt.show()