import pandas as pd
from sklearn import preprocessing

# print(pd.__version__)

exam_data = pd.read_csv("../data/exams.csv")
# display all data
# print(exam_data)


# comparing means

math_average = exam_data["math score"].mean()
reading_average = exam_data["reading score"].mean()
writing_average = exam_data["writing score"].mean()

print("maths avg ", math_average)
print("reading avg ", reading_average)
print("writing avg ", writing_average)

# using scikit learn to standardize the data

exam_data[["math score"]] = preprocessing.scale(exam_data[["math score"]])
exam_data[["reading score"]] = preprocessing.scale(exam_data[["reading score"]])
exam_data[["writing score"]] = preprocessing.scale(exam_data[["writing score"]])

# print standardized code
# mean is almost zero and sd is nearly 1
# print(exam_data)

# convert categorical variables into numerical data
# meaningful only to us not the model
# use LabelEncode and transform method

le = preprocessing.LabelEncoder()
exam_data["gender"] = le.fit_transform(exam_data["gender"].astype(str))
# display
print(exam_data.head())
# get the range of unique values
print(le.classes_)

# we use pandas to convert this race/ethnicity column into one hot representation
one_hot = pd.get_dummies(exam_data["race/ethnicity"])
# print(one_hot)
# assign to dataset
# exam_data["race/ethnicity"] = one_hot

exam_data = pd.get_dummies(
    exam_data,
    columns=[
        "race/ethnicity",
        "lunch",
        "parental level of education",
        "test preparation course",
    ],
)
# display
print(exam_data.head())
