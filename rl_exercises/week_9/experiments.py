# SAMPLE EFFICIENCY

# Use_model false: 17-33-03
# Use_model true: 17-27-18

import numpy as np
import matplotlib.pyplot as plt

# Load CSV data
log_file_nomodel = "outputs/2025-06-29/17-33-03/checkpoints/eval_log.csv"
log_file_modelbased = "outputs/2025-06-29/17-27-18/checkpoints/eval_log.csv"

data_ppo = np.genfromtxt(log_file_nomodel, delimiter=',', names=True)
data_dyna = np.genfromtxt(log_file_modelbased, delimiter=',', names=True)

# Plot average return vs real steps
def plot_average_return(real_steps, returns, title):
  plt.figure(figsize=(8, 5))
  plt.plot(real_steps, returns, marker="o", linestyle="-", label="Return")
  plt.xlabel("Real Steps")
  plt.ylabel("Avg Return")
  plt.title(title)
  plt.grid(True)
  plt.legend()
  plt.tight_layout()
  plt.show()

plot_average_return(data_dyna["real_steps"], data_dyna["mean_r"], title='Return vs Real Steps - Dyna-PPO')
plot_average_return(data_ppo["real_steps"], data_ppo["mean_r"], title='Return vs Real Steps - Plain PPO')

# How many steps sooner does Dyna-PPO reach 80 % of the final PPO return?
# Dyna-PPO reaches 80% of final PPO return 700.0 steps sooner.
final_ppo_return = data_ppo["mean_r"][-1]
threshold = 0.8 * final_ppo_return
ppo_cross = data_ppo["real_steps"][np.argmax(data_ppo["mean_r"] >= threshold)]
dyna_cross = data_dyna["real_steps"][np.argmax(data_dyna["mean_r"] >= threshold)]
step_difference = ppo_cross - dyna_cross
print(ppo_cross)
print(dyna_cross)
print(f"Dyna-PPO reaches 80% of final PPO return {step_difference} steps sooner.")

# Comparison of early returns: Is there an initial “model learning penalty” (Dyna underperforms early)?
# Early PPO avg return: 121.9046666666667; Early Dyna avg return: 149.38333333333333; Penalty: -27.478666666666626
# Only slight difference in performance in the first episodes
early_ppo = np.mean(data_ppo["mean_r"][data_ppo["mean_r"] <= 2000])
early_dyna = np.mean(data_dyna["mean_r"][data_dyna["mean_r"] <= 2000])

print(f"Early PPO avg return: {early_ppo}")
print(f"Early Dyna avg return: {early_dyna}")
print(f"Penalty: {early_ppo - early_dyna}")


