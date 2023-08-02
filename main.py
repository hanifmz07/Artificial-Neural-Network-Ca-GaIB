from src.ANN import ANN
from src.utils import *
import argparse
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()

parser.add_argument(
    "-d", "--data", default="bcancer.csv", type=str, help="CSV file directory"
)
parser.add_argument(
    "--epochs", type=int, default=1000, help="Amount of epochs in training ANN"
)
parser.add_argument(
    "--lr", type=float, default=0.1, help="Learning rate for parameter update"
)
parser.add_argument(
    "--batch_size", type=int, default=5, help="Amount of samples per batch"
)

args = parser.parse_args()

if args.data == "bcancer.csv":
    X, y = preprocess_bcancer()
else:
    print("Data not available")
    quit()

layer = [30, 15, 1]
model = ANN(
    layer_info=layer,
    learning_rate=args.lr,
    epochs=args.epochs,
    batch_size=args.batch_size,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
X_train, y_train = adjust_dimension(X_train, y_train)
X_test, y_test = adjust_dimension(X_test, y_test)

costs = model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"F1-score = {f1_score(np.squeeze(y_test), y_pred, show_y=True)}")
