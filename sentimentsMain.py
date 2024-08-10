# from pandasgui import show -- currently not using the show func nevertheless, you could obv
from PredictionModel import *
from AnalyzingInfo import *
from PreprocessingTxt import *
import pandas as pd
# from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


sentimentsData = pd.read_csv("Sentiment Dataset.csv")
sentimentsData = data_fixing(sentimentsData)
sentimentsData["text"] = lemmatize_text_column(sentimentsData.text)
text_data = handle_txt(sentimentsData)
text_data = pre_process_text(text_data)
text_data = lemmatize_data(text_data)

split_data = SplitData(sentimentsData["text"], sentimentsData["label"],
                       1)  # be using the 1 for the x val cus atm not using the x test
X_train, X_test, y_train, y_test = split_data.splitting_data()

# boolean_clf = BooleanModel() -- not using the boolean classifier cus it takes much more time to predict and has much worse accuracy
# boolean_clf.fit(X_train, y_train, text_data)
# predicted_boolean_labels = boolean_clf.predict(X_test)
# print(set(predicted_boolean_labels))

distances_clf = DistanceModel()
distances_clf.fit(X_train, y_train, text_data)

"""" 
unnecessary atm 👇
"""
# predicted_distances_labels = distances_clf.predict(X_test)
# print("accuracy: {:.2f}, precision: {:.2f}, f1: {:.2f}, recall: {:.2f}".format(
#     accuracy_score(y_test, predicted_distances_labels),
#     precision_score(y_test, predicted_distances_labels, average='macro'),
#     f1_score(y_test, predicted_distances_labels, average='macro'),
#     recall_score(y_test, predicted_distances_labels, average='macro')
# ))
#
# sentiments = ["sad", "joy", "love", "anger", "fear", "surprise"]
#
# while ((txt := input("insert a valid textual input in order to analyze your sentiments: \n")) != "-1"):
#     print(sentiments[(distances_clf.predict([txt]))[0]])
#

obtain_model = lambda: distances_clf
