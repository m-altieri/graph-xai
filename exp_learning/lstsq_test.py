import numpy as np

x1 = np.linspace(0.0, 0.5, 500, endpoint=False)
x2 = np.linspace(1.0, 0.5, 500, endpoint=False)[::-1]
x = np.concatenate((x1, x2))

y1 = np.full_like(x1, 0.0)
y2 = np.full_like(x2, 1.0)
y = np.concatenate((y1, y2))

A = np.vstack([x, np.ones(len(x))]).T
print(f"A: \n{A}")
print(f"y: \n{y}")

m, c = np.linalg.lstsq(A, y, rcond=None)[0]  # Slope (m) and y-intercept (c)

# Output the linear model
print("Linear model: y = {:.2f}x + {:.2f}".format(m, c))
