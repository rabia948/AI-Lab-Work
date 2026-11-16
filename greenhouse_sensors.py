import numpy as np

np.random.seed(7)
readings = np.random.randint(20, 41, size=(6, 3))
print(readings)

mean = readings.mean(axis=0)
print("Mean per sensor:", mean)

normalized = (readings - mean) / readings.std(axis=0)
print("Normalized column means:", np.round(normalized.mean(axis=0), 2) + 0.0)

print("Hour of max per sensor:", readings.argmax(axis=0))

with_mean = np.vstack([readings, mean])
print(with_mean.shape)

np.random.seed(7)
again = np.random.randint(20, 41, size=(6, 3))
print("Reproducible:", np.array_equal(readings, again))