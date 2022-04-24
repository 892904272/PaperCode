import matplotlib.pyplot as plt
import numpy as np
# new_ticks = np.linspace(10,100,9)
# print(new_ticks)
# plt.xticks(new_ticks)
#x=(10,20,30,40,50,60,70,80,90,100)
x=np.linspace(10,30,5)

y=[0.72,0.575,0.45,0.33,0.29]
y2=(0.735,0.62,0.475,0.364,0.35)
y3=(0.27,0.19,0.14,0.09,0)
y4=(0.2,0.08,0.05,0.03,0)
y5=(0.16,0.05,0.01,0,0)

plt.xlim(10,30)
# plt.ylim(0,1)
plt.plot(x,y,color="blue",marker="o",label="local")
plt.plot(x,y2,color="red",marker="x",label="offload")
plt.plot(x,y3,color="black",marker="D",label="DQN")
plt.plot(x,y4,color="gray",marker="H",label="DDPG")
plt.plot(x,y5,color="green",marker="*",label="PER-DDPG")
plt.xlabel("diuqi probability",fontproperties="SimSun")
plt.ylabel("diuqi and deadlinen",fontproperties="SimSun")

plt.legend()
plt.savefig("diuqi and deadline .png",dpi=500,bbox_inches="tight")
plt.show()