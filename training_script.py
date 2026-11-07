import glob
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

def load_training_datasets(pattern):
  file_list = glob.glob(pattern)
  if not file_list:
    print(
        f"Not found {pattern}"
    )
    return None

  df_list = [pd.read_csv(file) for file in file_list]
  merged_df = pd.concat(df_list, ignore_index=True)
  return merged_df



df = load_training_datasets("gestureRecognition/gestures_dataset_*.csv")

X = df.drop(columns=['label'])
y = df['label']
print("row counts:")
print(df["label"].value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 80/20 split, lets try LOO later, it gave tad better results i heard



#classifier = KNeighborsClassifier(n_neighbors=3) #change algo here, k is a crux of this algo, picking best one is *ART*
classifier = RandomForestClassifier(n_estimators=100)

# knn - gets closest matches "if im similar to x I must be x"
# random forest - gets label by multiple votes from decision trees 
# decision tree - splits decision into series of yes/no answers "if I sound like a duck, look like a duck, weight as a duck, I must be a duck"



classifier.fit(X_train, y_train)  # - MAGIC HERE, it learns, im baffled :O


y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print("\nRaport:\n", report)
print(f"\nAccuracy: { accuracy* 100:.2f}%") 

import joblib
joblib.dump(classifier, "models/gesture_model.pkl")