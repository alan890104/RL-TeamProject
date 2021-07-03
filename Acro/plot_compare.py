# Draw a graph with maxQ(s,a) on the first state s
#%%
import numpy as np
import matplotlib.pyplot as plt
import os

folder_names = ['original','nn1','Norm']

def polt_EWMA_compare():
    
    plt.figure(figsize = (10,5))
    plt.title('EWMA compare collected over time (Acrobot)')
    plt.xlabel('Episode')
    plt.ylabel('EWMA Score')
    length = 0

    for folder_name in folder_names:
        IQN_EWMA = []

        for filename in os.listdir(folder_name+"/Data/"):
            if 'EWMA0' in filename:
                with open(folder_name+'/Data/'+filename, 'r') as f:
                    IQN_EWMA.append(list(map(float,f.read().split())))
                

        IQN_EWMA = np.array(IQN_EWMA).transpose()
        IQN_EWMA_avg = np.mean(IQN_EWMA,axis=1)
        # IQN_std = np.std(IQN_EWMA,axis=1)
        length = max(length, len(IQN_EWMA_avg))
        
        plt.plot(IQN_EWMA_avg ,label='IQN({})-{}'.format(folder_name,len(IQN_EWMA_avg)), alpha = 0.8)
        
    # optimal_value = 195
    # plt.plot([optimal_value for _ in range(length)],'r--',label='optimal-score', alpha = 0.5)
    plt.legend(loc='best')
    plt.savefig('PIC/Acrobot-compare.png')
    plt.show()

if __name__=='__main__':
    try:
        polt_EWMA_compare()
    except Exception as e:
        print(e)