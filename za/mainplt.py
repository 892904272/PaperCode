#开发人员： 诺痕
#开发时间： 2021-12-06 9:49
import matplotlib.pyplot as plt
import numpy as np
# new_ticks = np.linspace(10,100,9)
# print(new_ticks)
# plt.xticks(new_ticks)
#x=(10,20,30,40,50,60,70,80,90,100)
x=np.linspace(10,100,10)
# y=[0.313725490196078,0.253164556962025,0.247238742565845,0.23992443324937,0.268256333830104,0.252572898799314,0.243709561466571,0.240424847119408,0.267384105960265,0.254067584480601]
# y2=(0,0,0,0,0,0.006846384,0.00035524,0.000906618313689937,0.00356066831005204,0.004638671875)
# y3=(0.242574257425743,0.230964467005076,0.26062091503268,0.252336448598131,0.227944926058134,0.296818364881192,0.227889908256881,0.235161290322581,0.210600507471102,0.245843828715365)
#plt.scatter(x,y,label="test")
# y=[1-0.313725490196078,1-0.253164556962025,1-0.247238742565845,1-0.23992443324937,1-0.268256333830104,1-0.252572898799314,1-0.243709561466571,1-0.240424847119408,1-0.267384105960265,1-0.254067584480601]
# y2=(1-0,1-0,1-0,1-0,1-0,1-0.006846384,1-0.00035524,1-0.000906618313689937,1-0.00356066831005204,1-0.004638671875)
# y3=(1-0.242574257425743,1-0.230964467005076,1-0.26062091503268,1-0.252336448598131,1-0.227944926058134,1-0.296818364881192,1-0.227889908256881,1-0.235161290322581,1-0.210600507471102,1-0.245843828715365)
y=[0.863861386,0.842038217,0.840890354492993,0.842302878598248,0.831713554987212,0.847583643122677,0.845984598459846,0.844151771715271,0.841260906276386,0.849089548515839]
y2=(0.425301204819277,0.414285714285714,0.432477216238608,0.456535737282679,0.49975597852611,0.506849315068493,0.498904309715121,0.55893300248139,0.596734693877551,0.623744527427247)
y3=(0.174581280788177,0.190284360189573,0.208156329651657,0.222503160556258,0.27463731865933,0.283777871266302,0.33522929257021,0.34522929257021,0.35522929257021,0.38522929257021)

plt.xlim(10,100)
plt.ylim(0,1)
plt.plot(x,y,color="blue",marker="o",label="local offloading")
plt.plot(x,y2,color="red",marker="x",label="random")
plt.plot(x,y3,color="black",marker="D",label="Double-DQN union Dueling-DQN")
plt.xlabel("用户设备数",fontproperties="SimSun")
plt.ylabel("未完成任务率",fontproperties="SimSun")

plt.legend()
plt.savefig("未完成任务率.png",dpi=500,bbox_inches="tight")
plt.show()