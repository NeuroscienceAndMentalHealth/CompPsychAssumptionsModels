# Online-TestRetest

Simple models for the following behavioural tasks:

 - 4 Arm bandit task
 - Effort task
 - Gambling task
 - Reward bias task 
 - Exploration task

# Contents

    .
    ├── BanditTask         # Four-armed bandit task
    │   └── models       
    │         ├── Model_1  # NULL model (Parameters: Temperature)
    │         ├── Model_2  # RL model (Parameters: Learning rate, Temperature)
    │         ├── Model_3  # RL model (Parameters: Learning rate punishments, Learning rate rewards, Temperature)
    │         ├── Model_4  # RL model (Parameters: Learning rate punishments, Learning rate rewards, Reward sensitivity, Punishment sensitivity, Lapse)
    │         └── Model_5  # RL model (Parameters: Learning rate negative, Learning rate positive, Decay negative, Decay positive, Temperature)
    │    
    ├── EffortTask         # Cognitive Effort task
    │   └── models       
    │         ├── Model_1  # NULL model (Parameters: Intercept)
    │         ├── Model_2  # Logistic model (Parameters: Intercept, Linear Reward, Linear Effort)
    │         ├── Model_3  # Logistic model (Parameters: Intercept, Linear Reward, Linear Effort, Quadratic Effort)
    │         ├── Model_4  # Robust Logistic model (Parameters: Intercept, Linear Reward, Linear Effort, Lapse)
    │         └── Model_5  # Robust Logistic model (Parameters: Intercept, Linear Reward, Linear Effort, Quadratic Effort, Lapse)
    │    
    ├── GamblingTask       # Gambling task
    │    └── models       
    │         ├── Model_1  # NULL model (Parameters: Temperature)
    │         ├── Model_2  # Risk Aversion model (Parameters: Risk Aversion, Temperature)
    │         ├── Model_3  # Loss Aversion model (Parameters: Loss Aversion, Temperature)
    │         ├── Model_4  # Risk & Loss Aversion model (Parameters: Risk Aversion, Loss Aversion, Temperature)
    │         └── Model_5  # Sep. Risk & Loss Aversion model (Parameters: Risk Aversion gain trials, Risk Aversion loss trials, Loss Aversion, Temperature)
    │
    ├── ExplorationTask    # Explore task
    │    └── models       
    │         ├── Model_1  # NULL model (Parameters: Intercept)
    │         ├── Model_2  # Value Difference model (Parameters: Theta_V)
    │         ├── Model_3  # Weighted Value difference model (Parameters: Theta_V/TU)
    │         ├── Model_4  # Value difference + sigma difference model (Parameters: Theta_V, Theta_RU)
    │         └── Model_5  # Hybrid model (Parameters: Theta_V, Theta_RU, Theta_V/TU)
    │
    └── RewardBiasTask     # Reward bias task/ Pizzagalli
        └── models       
              ├── Model_1  # NULL model (Parameters: Temperature)
              ├── Model_2  # Win-stay lose-shift model (Parameters: Temperature)
              ├── Model_3  # Stimulus-action model (Parameters: Learning rate, Reward sensitivity, Instruction sensitivity, Initial q-value)
              ├── Model_4  # Stimulus-action model with separate sensitivities (Parameters: Learning rate, Reward sensitivity, Punishment sensitivity, Instruction sensitivity, Initial q-value)
              └── Model_5  # Action only model (Parameters: Learning rate, Reward sensitivity, Instruction sensitivity, Initial q-value - same parameters as M3 but diff meanings)

