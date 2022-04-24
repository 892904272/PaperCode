bendi=42
xiezai = 135
daxiao = 8
total =xiezai


bendiEnergy = daxiao*0.775*bendi*0.9


xiezaiEnergy = total*(daxiao/14)*1.3+xiezai*(daxiao*0.1)*10

print("bendi:"+str(bendiEnergy))
print("xiezai:"+str(xiezaiEnergy))
print("he:"+str(bendiEnergy+xiezaiEnergy))