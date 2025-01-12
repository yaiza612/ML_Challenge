import numpy as np
a = np.random.random(size=(10000, 2))
b = np.random.random(size=(10000, 2))
c = np.random.random(size=(10000, 2))
preds = [a, b, c]
new_stacked_predictions = np.stack([p for p in preds])
summed_predictions = np.sum(new_stacked_predictions, axis=0)
print(summed_predictions.shape)
print(new_stacked_predictions.shape)

z = np.load("notebooks/stack_test_preds_4_try_final_stacknet.npy")
y = np.load("notebooks/stack_test_preds_4_for_test_predictions_w_restacking.npy")
x = np.load("notebooks/stack_test_preds_4_rf.npy")
a = 2
