'''Contains file for creating Gradient Descent plots.'''
from gradient_descent import grad_desc, stoch_grad_desc
import matplotlib.pyplot as plt
import time

# Initializing arrays for returns
start_time1 = time.time()
loss, slope, intrcpt = grad_desc()
end_time1 = time.time()
start_time2 = time.time()
loss2, slope2, intrcpt2 = stoch_grad_desc()
end_time2 = time.time()

# Getting run times for both versions
print("Time for Gradient Descent to run: " + str(end_time1 - start_time1))
print("Time for Stochastic Gradient Descent to run: "
      + str(end_time2 - start_time2))

# Plots for Gradient Descent and Stochastic version
t = list(range(1, 1001, 1))
# Loss plot
plt.figure()
plt.plot(t, loss, label="Gradient Descent")
plt.plot(t, loss2, label="Stochastic Gradient Descent")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.suptitle("Loss Function Values over 1000 iterations")
plt.title("Stochastic vs Gradient Descent")
plt.show()
plt.savefig("1000iterLossPlot.png")
plt.clf

# Slope plot
plt.figure()
plt.plot(t, slope, label="Gradient Descent")
plt.plot(t, slope2, label="Stochastic Gradient Descent")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Slope")
plt.suptitle("Slope Values over 1000 iterations")
plt.title("Stochastic vs Gradient Descent")
plt.show()
plt.savefig("1000iterSlopePlot.png")
plt.clf

# Intercept plot
plt.figure()
plt.plot(t, intrcpt, label="Gradient Descent")
plt.plot(t, intrcpt2, label="Stochastic Gradient Descent")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Intercept")
plt.suptitle("Intercept Values over 1000 iterations")
plt.title("Stochastic vs Gradient Descent")
plt.show()
plt.savefig("1000iterIntrPlot.png")
plt.clf
