#%%
'''
REFERENCE:
https://datascience.stackexchange.com/questions/40874/how-does-implicit-quantile-regression-network-iqn-differ-from-qr-dqn
https://github.com/deligentfool/dqn_zoo/blob/master/IQN/iqn.py
https://zhuanlan.zhihu.com/p/60949506
https://blog.csdn.net/clover_my/article/details/90777964
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import gym
import math
from datetime import datetime, time
from collections import deque
from scipy.stats import norm
import os

class replay_buffer(object):
    '''
    FUNCTION
    ------
    1. store(state,action,reward,next_state,done)
    2. sample(batch_size)  
    '''
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)
    
    def store(self,state,action,reward,next_state,done):
        self.memory.append([state,action,reward,next_state,done])
    
    def sample(self,batch_size):
        batch =  random.sample(self.memory, batch_size)
        observation, action, reward, next_observation, done = zip(*batch)
        return observation, action, reward, next_observation, done
    
    def __len__(self):
        return len(self.memory)


class net(nn.Module):
    '''   
    input :
    ------- 
      torch.tensor([[0],[1],[2].....])  
    output: 
    -------
      Z_value,tau  
    steps:  
    ------
    1. takes a current state S convert to V
    2. generate "1" scalar τ~U[0,1]
    3. expand |τ| into 64 by cos(πiτ) i=0~63
    4. feed τ into function ϕ(τ), get H  
       where ϕj = ReLU( Σ cos(πiτ)wij + bj )
       (use neural network to get w and b)
    5. V ⊙ H (V and H have same dimension)
    6. run 2~4 in parallel(treat as a batch)
    7. pass to fully connected layer
    8. return Z value and tau
    '''
    def __init__(self, n_states=4, n_actions=2, quant_num = 64):
        super(net, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.quant_num = quant_num

        # for state S to V
        self.convert_to_V = nn.Linear(4,128) 

        # ϕ for weight and bias
        self.phi = nn.Linear(1,128) 
        self.phi_bias = nn.Parameter(torch.zeros(128),requires_grad=True)

        # fully connected layer
        self.fc1 = nn.Linear(128,128)
        self.fc2 = nn.Linear(128,self.n_actions)

    def forward(self,state):
        # for state S to V
        V = F.relu(self.convert_to_V(state)) # V.size() = [1,256]

        # ϕ for weight and bias
        tau = torch.rand(self.quant_num,1) # τ.size() = [64,1] (選64個τ)
        # tau = self.Wang(tau,0.75)
        quants = torch.arange(0,self.quant_num,1.0) # quant.size() = [64]
        cos_trans = torch.cos(quants*tau*np.pi).unsqueeze(2).cuda() # cos_trans.size() = [64,64,1]
        H = F.relu(self.phi(cos_trans).sum(1)+self.phi_bias.unsqueeze(0)).unsqueeze(0)# H.size() = [1,64,256]
        V = V.unsqueeze(1)

        # elementwise product
        x = V*H

        # fully connected layer
        x = F.relu(self.fc1(x)) 
        Z_value = self.fc2(x).transpose(1, 2) # Z_value.size() = [batch,166,64]
        tau = tau.squeeze().unsqueeze(0).expand(x.size(0), self.quant_num) # tau.size() = [batch,64]

        return Z_value,tau

    def Pow(self,tau,scale):
        if scale>=0:
            return tau**(1/(1+abs(scale)))
        else:
            return 1-(1-tau)**(1/(1+abs(scale)))
    
    def Wang(self,tau,scale):
        return torch.Tensor(norm.cdf(norm.ppf(tau)+scale)).float()
    
    def CPW(self,tau,scale):
        tmp = tau**scale
        inv = (1-tau)**scale
        return (tmp+inv)**(1/scale)
    
    def CVaR(self,tau,scale):
        return tau*scale

class IQN(object):
    def __init__(self, env=None, capacity=10000, episode=1000, exploration=1000, k_sample=8, k=1, n=1, n_prime=1, gamma=0.9, batch_size=32, epsilon=0.05, learning_rate=1e-3, quant_num=64, update_freq=100, render=True):
        self.env = env
        self.capacity = capacity # buffer capacity
        self.episode = episode
        self.exploration = exploration # when to learning
        self.k = k # kappa in the papper, to do quantile huber loss
        self.k_sample = k_sample # take avarage over k_sample forwarding to evaluate the best action
        self.n = n # n and n' forward
        self.n_prime = n_prime
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.quant_num = quant_num
        self.update_freq = update_freq
        self.render = render
        self.count = 0


        self.n_states = 4
        self.n_actions = 2
        self.net = net(self.n_states, self.n_actions,quant_num=self.quant_num).cuda()
        self.target_net = net(self.n_states, self.n_actions,quant_num=self.quant_num).cuda()
        self.target_net.load_state_dict(self.net.state_dict())
        self.buffer = replay_buffer(self.capacity)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.epsilon = epsilon
        self.step = 0

    def save_model(self,complete=False):
        today = datetime.now().strftime("%Y-%m-%d_%H-%M")
        torch.save(self.net.state_dict(), 'pkl/{}_net_pred.pkl'.format(today))
        torch.save(self.target_net.state_dict(),'pkl/{}_target_net.pkl'.format(today))
    
    def load_model(self):
        self.net.load_state_dict(torch.load('pkl/net_pred.pkl'))
        self.target_net.load_state_dict(torch.load('pkl/target_net.pkl'))

    def decay(self):
        # if self.epsilon>0.05: self.epsilon*=0.99
        self.epsilon*=0.99
    
    def choose_action(self,state):
        '''  
        concept:  
          more chance to exploration new state at the beginning  
          to be conservative at the end  
        algorithm:  
          if random bigger than epsilon:  
              forward K time and get average Z value  
              and then pick the argmax of it
          else:  
              random choose from action space
        '''
        if random.random()>self.epsilon: # initail epsilon = 1.0
            total_value = 0 # total_value will be resized to [batch,166,64]
            for _ in range(self.k_sample):
                Z_value, _ = self.net.forward(state) # state must be 2d ex: [[0],[1],[2],[3]]
                total_value += Z_value
            action = total_value.mean(2).max(1)[1].detach().item()
        else:
            action = random.choice(range(self.n_actions))
        
        return action

    def compute_loss(self,taus,values,target_values):
        # quantile huber loss
        '''
        quantile loss:  
        -------------
        residual = y_target - y_pred  
        loss = max(quantile * residual, (quantile - 1) * residual)
        huber loss:
        -----------  
        for all x in error (error = y_target - y_pred )  
        loss = 0.5 * x^2                  if |x| <= d  
        loss = 0.5 * d^2 + d * (|x| - d)  if |x| > d
        '''
        loss = 0
        for value, tau in zip(values,taus):
            for target_value in target_values:
                error = target_value - value
                huber_loss = 0.5*error.abs().clamp(min=0,max=self.k).pow(2)
                huber_loss += self.k*( error.abs() - error.abs().clamp(min=0.,max=self.k) - 0.5*self.k)
                quantile_loss = (tau-(error<0).float()).abs() * huber_loss
                loss += quantile_loss.sum()/self.batch_size
        return loss

    def learn(self):
        '''
        steps:  
        ------
        1. update target net by net every 200 times  
        2. sample 32 memories 
        3. pass memory to net and target net  
        4. compute loss by quantile huber loss
        5. zero grad
        6. back propagation
        7. optimize by Adam  
        '''
        if self.count % self.update_freq == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        states,actions,rewards,next_states,dones = self.buffer.sample(self.batch_size)
        states = torch.Tensor(states).cuda()
        actions = torch.LongTensor(actions).cuda()
        rewards = torch.Tensor(rewards).unsqueeze(1).expand(self.batch_size, self.quant_num).cuda()
        next_states = torch.Tensor(next_states).cuda()
        dones = torch.Tensor(dones).unsqueeze(1).expand(self.batch_size, self.quant_num).cuda() # dones[32] -> dones[32,64]

        values        = []
        taus          = []
        target_values = []

        for _ in range(self.n):
            Z_value,tau = self.net.forward(states) # convert scalar to [[1],[2],[3]...]
            #print("HELPING",Z_value.size())
            value = Z_value.gather(1, actions.unsqueeze(1).unsqueeze(2).expand(self.batch_size, 1, self.quant_num)).squeeze()
            values.append(value.cpu())
            taus.append(tau)

        for _ in range(self.n_prime):
            target_Z_value, target_tau = self.target_net.forward(next_states)
            target_actions = target_Z_value.sum(2).max(1)[1].detach()
            target_value = target_Z_value.gather(1, target_actions.unsqueeze(1).unsqueeze(2).expand(self.batch_size, 1, self.quant_num)).squeeze()
            target_value =  rewards + self.gamma * target_value *(1. - dones)
            target_value = target_value.detach()
            target_values.append(target_value.cpu())


        loss = self.compute_loss(taus,values,target_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        
        # self.decay() # decay epsilon and learning rate 

    def train(self,ID):
        '''
        steps:
        ------
        1. reset env  
        2. done = false  
        3. if done is true, go to step 9
        4. choose best action 
        5. get feedback from env  
        6. store memory  
        7. if memory bigger than "exploration" then "learn"  
        8. go to next state
        9. record the maximum of Z_value from target net
        '''
        print('start'+str(ID))
        EWMA_reward = None
        f1 = open('Data/CartPole-v0_EWMA' + str(ID) + '.txt', 'w+')
        f2 = open('Data/CartPole-v0_score' + str(ID) + '.txt', 'w+')
        for i in range(self.episode):
           
            state = self.env.reset()
            # if self.render: self.env.render()
            score = 0
            win = 0
            while True:
                action = self.choose_action(torch.Tensor([state]).cuda())
                next_state,reward,done,_ = self.env.step(action)
                score += reward
                # if self.render: self.env.render()
                next_state = np.array([0,0,0,0]) if next_state is None else next_state 
                self.buffer.store(state,action,reward,next_state,done)
                self.count += 1
                if self.count > self.exploration: self.learn()
                if done:    break
                state = next_state
            EWMA_reward = score if EWMA_reward is None else EWMA_reward*0.94 + score*0.06
            print('episode: {}\t score: {}\t EWMA_reward: {}'.format(i, score, EWMA_reward))
            f1.write(str(EWMA_reward)+'\n')
            f2.write(str(score)+'\n') 

            if EWMA_reward >= 195:
                break
        f1.close()
        f2.close()
        self.save_model()
        print('finish'+str(ID))
    
    def test(self,path,epoch=1):
        NET = net().cuda()
        NET.load_state_dict(torch.load(path))
        Expected_reward = 0
        for _ in range(epoch):
            state = self.env.reset()
            total_reward = 0
            while True:
                self.env.render()
                #========================choose action==============================
                total_value = 0
                for _ in range(self.k_sample):
                    Z_value, _ = NET.forward(torch.Tensor([state]).cuda())
                    total_value += Z_value
                action = total_value.mean(2).max(1)[1].detach().item()
                #========================choose action==============================
                next_state,reward,done,_ = self.env.step(action)
                total_reward += reward
                if done:    break
                state = next_state
            Expected_reward += total_reward
        print('Testing for {} times, the Expected Reward is {}'.format(epoch,Expected_reward/epoch))
        self.env.close()

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def main(ID):
    env = gym.make('CartPole-v0')
    rand_seed = 20
    env.seed(rand_seed)
    seed_torch(rand_seed)
    print('observation_space : ',env.observation_space)
    print('action_space : ',env.action_space)
    a = IQN(env,episode=2000,epsilon=0.05,learning_rate=1e-4,gamma=0.95,quant_num=64)
    a.train(ID)
    # a.test(path=r"pkl\2021-03-16_05點47分_target_net.pkl")

if __name__ == '__main__':
    for i in range(0,1):
        main(i)

