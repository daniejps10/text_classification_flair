import constants

from flair.models import TextClassifier
from flair.data import Sentence

# Plot Loss Curve
from flair.visual.training_curves import Plotter

# Making Prediciton
# Load saved model and predict
print(constants.SUB_LINE)
print('Load saved model and predict...')
new_clf = TextClassifier.load(constants.INPUT_FOLDER_FILE.format('best-model.pt'))

# Sample Sentence
print(constants.SUB_LINE)
print('Sentences of sample...')
ex1 = Sentence("That girl is a bitch")
ex2 = Sentence("This is a good material")

# Apply our model
print(constants.SUB_LINE)
print('Make predictions...')
new_clf.predict(ex1)
new_clf.predict(ex2)
print(ex1.labels)
print(ex2.labels)

# Plot Loss Curve
print(constants.SUB_LINE)
print('Plot Loss Curve...')
plotter = Plotter()
plotter.plot_training_curves(constants.INPUT_FOLDER_FILE.format('loss.tsv'))
plotter.plot_weights(constants.INPUT_FOLDER_FILE.format('weights.txt'))