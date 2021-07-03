# Draw a graph with maxQ(s,a) on the first state s
#%%
import numpy as np
import matplotlib.pyplot as plt
import os

folder_name = 'Wang75'

def polt_EWMA_score():
    IQN_EWMA = []
    IQN_score = []
    
    for filename in os.listdir(folder_name+"/Data/"):
        if 'EWMA0' in filename:
            with open(folder_name+'/Data/'+filename, 'r') as f:
                IQN_EWMA.append(list(map(float,f.read().split())))
        if 'score0' in filename:
            with open(folder_name+'/Data/'+filename, 'r') as f:
                IQN_score.append(list(map(float,f.read().split())))
            

    IQN_EWMA = np.array(IQN_EWMA).transpose()
    IQN_EWMA_avg = np.mean(IQN_EWMA,axis=1)
    # IQN_std = np.std(IQN_EWMA,axis=1)

    IQN_score = np.array(IQN_score).transpose()
    IQN_score_avg = np.mean(IQN_score,axis=1)



    plt.figure(figsize = (10,5))
    plt.title('Score and EWMA collected over time (PendulumEnv-{})'.format(folder_name))
    plt.xlabel('Episode')
    plt.ylabel('Score')

    optimal_value = 0

    plt.plot([optimal_value for _ in range(len(IQN_EWMA_avg))],'r--',label='optimal-score')
    
    plt.plot(IQN_score_avg, 'green',label='IQN-score', alpha = 0.3)
    plt.plot(IQN_EWMA_avg, 'blue',label='IQN-EWMA')
    
    # plt.fill_between([i for i in range(len(IQN_EWMA_avg))], IQN_EWMA_avg+IQN_std, IQN_EWMA_avg-IQN_std, alpha = 0.3)

    plt.legend(loc='best')
    plt.savefig('PIC/{}.png'.format(folder_name))
    plt.show()

if __name__=='__main__':
    try:
        polt_EWMA_score()
    except Exception as e:
        print(e)