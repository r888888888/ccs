import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from pprint import pprint

def _tag_tokenizer(x):
  return x.split(" ")

_MIN_DF = 0.01
_MAX_DF = 0.5
data = pd.read_csv("~/Development/tf-data/dataset/posts.csv")
cv = CountVectorizer(min_df=_MIN_DF, max_df=_MAX_DF, tokenizer=_tag_tokenizer)
cv.fit(data["tags"])
tags = set(cv.vocabulary_.keys())
pprint(tags)
print(len(tags))