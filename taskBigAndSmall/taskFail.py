import matplotlib.pyplot as plt
import numpy as np
# new_ticks = np.linspace(10,100,9)
# print(new_ticks)
# plt.xticks(new_ticks)
#x=(10,20,30,40,50,60,70,80,90,100)
x=np.linspace(1,10,10)
#任务丢弃率
y=[0,0,0.0143387,0.12,0.280112,0.313131,0.498,0.6353,0.6794,0.7427]
y2=(0,0,0,0.0543,0.207,0.275,0.44,0.591,0.66,0.73112)
y3=(0,0,0,0,0.018867,0.06,0.1164,0.32,0.384,0.5323)
y4=(0,0,0,0,0,0.023,0.06533,0.1256,0.1543,0.321)
y5=(0,0,0,0,0,0.012433,0.053533,0.102156,0.1243,0.2821)

plt.xlim(1,10)
# plt.ylim(0,1)
plt.plot(x,y,color="blue",marker="o",label="local")
plt.plot(x,y2,color="red",marker="x",label="offload")
plt.plot(x,y3,color="black",marker="D",label="DQN")
plt.plot(x,y4,color="gray",marker="H",label="DDPG")
plt.plot(x,y5,color="green",marker="*",label="PER-DDPG")
plt.xlabel("task arrival probability",fontproperties="SimSun")
plt.ylabel("Energy consumption",fontproperties="SimSun")

plt.legend()
plt.savefig("Energy consumption.png",dpi=500,bbox_inches="tight")
plt.show()