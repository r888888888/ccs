#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from pprint import pprint

def _tag_tokenizer(x):
  return x.split(" ")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

_MIN_DF = 150
_MAX_DF = 0.5
_IGNORE_TAGS = set(["absurdres", "highres", "character_name", "character_request", "commentary", "commentary_request", "copyright_name", "official_art", "translation_request", "translated", "transparent_background", "twitter_username", "1boy", "2boys", "1girl", "2girls", "3girls", "simple_background", "white_background"])
data = pd.read_csv(os.path.normpath("/var/lib/ccs/data/dataset/posts_chars.csv"))
cv = CountVectorizer(min_df=_MIN_DF, max_df=_MAX_DF, tokenizer=_tag_tokenizer)
cv.fit(data["tags"])
tags = set(cv.vocabulary_.keys()) - _IGNORE_TAGS
rows = data["tags"][data["tags"].isin(tags)]
print("total images", len(rows))
print("total tags", len(tags))
pprint(rows.value_counts())
