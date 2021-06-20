# Wumpus_Q-Learning (warm-up project!)
Implementation of 5*5 Wumpus game using Q-Learning algorithm with python and javascript 

There are 2 files, one is the implementation without graphics with python and the other one is implemented graphically with javascript.
I used simple Q-Learning formula. (see the equation below)
![formula](https://user-images.githubusercontent.com/85555218/122079965-237c9600-ce13-11eb-8c86-c5506ddd20c0.png)

    goal reward: 100
    holes and wall rewards = -100
    other states reward = -1
    
    discount factor: 0.8
    
    episodes = 500
    
### the following animation shows the performance of the Q-Learning algorithm with 100 episodes for training: (you can also see the changes of Q(s, a) per episode)
![wumpus](https://user-images.githubusercontent.com/85555218/122080202-56bf2500-ce13-11eb-8225-14a03dfd16fc.gif)
