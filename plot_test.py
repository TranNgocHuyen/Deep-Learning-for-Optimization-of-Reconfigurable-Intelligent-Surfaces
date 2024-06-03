import matplotlib.pyplot as plt
import torch
import numpy as np
# Thay đổi kích thước
plt.figure(figsize=(5,4))

y=[0.1,0.5,1]

samples_1e12_PV=torch.load("/home/tranngochuyen/Do_an/Deep-Learning-for-Optimization-of-Reconfigurable-Intelligent-Surfaces/results_server/30-05-2024_06-43-28/iter_WSR.pt")
#samples_1e12_PV=torch.load("/home/tranngochuyen/Do_an/Deep-Learning-for-Optimization-of-Reconfigurable-Intelligent-Surfaces/results_server/28-05-2024_12-00-06/iter_WSR.pt")
print(len(samples_1e12_PV)) # list, len= 201
x_1e12_PV=list(range(len(samples_1e12_PV)))
plt.plot(x_1e12_PV,samples_1e12_PV, 'm-^',label='TSNR=1e12,PV')



samples=torch.load("/home/tranngochuyen/Do_an/Deep-Learning-for-Optimization-of-Reconfigurable-Intelligent-Surfaces/results_server/26-05-2024_21-00-02/iter_WSR.pt")
#samples=torch.load("/home/tranngochuyen/Do_an/Deep-Learning-for-Optimization-of-Reconfigurable-Intelligent-Surfaces/results_server/28-05-2024_12-00-06/iter_WSR.pt")
print(len(samples)) # list, len= 201
x_1e12=list(range(len(samples)))
plt.plot(x_1e12,samples, 'b-^',label='TSNR=1e12,PI')


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