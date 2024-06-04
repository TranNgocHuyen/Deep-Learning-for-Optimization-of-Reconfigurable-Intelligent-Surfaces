import matplotlib.pyplot as plt
import torch
import numpy as np
# Thay đổi kích thước
plt.figure(figsize=(5,4))

# PV 
samples_1e11_PV=torch.load('/home/tranngochuyen/Do_an/Deep-Learning-for-Optimization-of-Reconfigurable-Intelligent-Surfaces/results_server/29-05-2024_17-03-40/iter_WSR.pt')
print(len(samples_1e11_PV)) # list, len= 201
x_1e11_PV=list(range(len(samples_1e11_PV)))
plt.plot(x_1e11_PV,samples_1e11_PV, 'b-',label='TSNR=1e11,PV') #x,y là list

#===================================================================
samples_5e11_PV=torch.load('/home/tranngochuyen/Do_an/Deep-Learning-for-Optimization-of-Reconfigurable-Intelligent-Surfaces/results_server/29-05-2024_22-26-58/iter_WSR.pt')
print(len(samples_5e11_PV)) # list, len= 201
x_5e11_PV=list(range(len(samples_5e11_PV)))
plt.plot(x_5e11_PV,samples_5e11_PV, 'g-',label='TSNR=5e11,PV') #x,y là list
#===================================================================
samples_1e12_PV=torch.load("/home/tranngochuyen/Do_an/Deep-Learning-for-Optimization-of-Reconfigurable-Intelligent-Surfaces/results_server/30-05-2024_06-43-28/iter_WSR.pt")
print(len(samples_1e12_PV)) # list, len= 201
x_1e12_PV=list(range(len(samples_1e12_PV)))
plt.plot(x_1e12_PV,samples_1e12_PV, 'm-',label='TSNR=1e12,PV')

#PI

samples_1e11=torch.load("/home/tranngochuyen/Do_an/Deep-Learning-for-Optimization-of-Reconfigurable-Intelligent-Surfaces/results_server/25-05-2024_20-00-18/iter_WSR.pt")
print(len(samples_1e11)) # list, len= 201
x_1e11=list(range(len(samples_1e11)))
plt.plot(x_1e11,samples_1e11, color='tab:orange',label='TSNR=1e11,PI') #x,y là list

#===================================================================
samples_5e11_PI=torch.load("/home/tranngochuyen/Do_an/Deep-Learning-for-Optimization-of-Reconfigurable-Intelligent-Surfaces/results_server/27-05-2024_23-19-24/iter_WSR.pt")
print(len(samples_5e11_PI)) # list, len= 201
x=list(range(len(samples_5e11_PI)))
plt.plot(x,samples_5e11_PI, 'r-',label='TSNR=5e11,PI') #x,y là list
#===================================================================
samples_1e12_PI=torch.load("/home/tranngochuyen/Do_an/Deep-Learning-for-Optimization-of-Reconfigurable-Intelligent-Surfaces/results_server/28-05-2024_12-00-06/iter_WSR.pt")
print(len(samples_1e12_PI)) # list, len= 201
x_1e12=list(range(len(samples_1e12_PI)))
plt.plot(x_1e12,samples_1e12_PI, 'tab:brown',label='TSNR=1e12,PI')

# thay đổi độ chia 
plt.xticks(np.arange(0,400,200))
plt.yticks(np.arange(0,2.5,0.5))

# Thay đổi kích thước

#plt.xlim(0,200)
#plt.ylim(0,3)
# Chú thích
plt.title('Improvement or WSR in traning')
plt.xlabel('Iteration')
plt.ylabel('WSR(bit/Hz/s)')
#plt.legend sẽ đi tìm các thành phần chứa tham số label và đưa vào chú thích 
plt.legend(loc="best") # add chủ thích

plt.grid(True)
plt.show()