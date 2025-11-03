import numpy as np
# inputs: y_true, y_pred
diff = y_true - y_pred
e = float(np.sqrt(np.mean(diff ** 2)))