"""Build the MLP bootcamp dataset (Craigslist used cars -> 4 price tiers).

Usage: python3 02_build_dataset.py <vehicles_price_tier.csv> <vehicles.csv (raw, has price)> <output_dir>

Outputs
  <out>/student/train.csv, test.csv, sample_submission.csv, descriptions.csv, README.md
  <out>/mentor/solution.csv, full_with_price.csv
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
TIER_BINS = [1000, 7995, 15995, 27995, 150000]

src_tier, src_raw, out = sys.argv[1], sys.argv[2], sys.argv[3]
student_dir, mentor_dir = os.path.join(out, "student"), os.path.join(out, "mentor")
os.makedirs(student_dir, exist_ok=True)
os.makedirs(mentor_dir, exist_ok=True)

df = pd.read_csv(src_tier, low_memory=False)
price = pd.read_csv(src_raw, usecols=["id", "price"])
df = df.merge(price, on="id", how="left", validate="one_to_one")
assert df.price.notna().all()
assert (pd.cut(df.price, TIER_BINS, labels=[1, 2, 3, 4], include_lowest=True).astype(int) == df.price_tier).all()

# ---------------------------------------------------------------- seller_type
df["seller_type"] = df.url.str.extract(r"/(cto|ctd)/")[0].map({"cto": "owner", "ctd": "dealer"})
assert df.seller_type.notna().all()

# ---------------------------------------------------------------- shuffle + new ID
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
df.insert(0, "listing_id", np.arange(len(df)))

# ---------------------------------------------------------------- repost groups
# Rows are linked if they share a VIN or an identical (year, make, model, odometer, description).
# Connected components give one group per physical car / ad.
desc = df.description.fillna("")
content_key = pd.util.hash_pandas_object(
    df[["year", "manufacturer", "model", "odometer"]].astype(str).assign(d=desc), index=False
)
keys = pd.concat([
    pd.Series("v_" + df.VIN.astype(str), index=df.index)[df.VIN.notna()],
    pd.Series("c_" + content_key.astype(str), index=df.index)[desc.str.len() > 0],
])
key_codes, _ = pd.factorize(keys)
n, k = len(df), key_codes.max() + 1
graph = coo_matrix((np.ones(len(keys)), (keys.index.values, n + key_codes)), shape=(n + k, n + k))
_, labels = connected_components(graph, directed=False)
df["group_id"] = pd.factorize(labels[:n])[0]

# ---------------------------------------------------------------- 80/20 grouped, stratified split
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
_, test_idx = next(sgkf.split(df, df.price_tier, groups=df.group_id))
df["split"] = "train"
df.loc[test_idx, "split"] = "test"
assert not set(df.group_id[df.split == "train"]) & set(df.group_id[df.split == "test"])

# ---------------------------------------------------------------- description masking
PRICE_WORDS = r"(?:asking|ask|price|priced|obo|firm|sell(?:ing)? for|sale for|for sale at|want|wanting|cash price|sale price|list price|msrp|offers? under|less than|lowest|take)"
RE_DOLLAR = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?\s?[kK]?\b|\$\s?\d")
RE_WORD_NUM = re.compile(PRICE_WORDS + r"(\W{0,15})(\d{1,3}(?:,\d{3})+|\d{3,6}|\d{1,3}(?:\.\d)?\s?[kK])\b", re.I)
RE_NUM_WORD = re.compile(r"\b(\d{1,3}(?:,\d{3})+|\d{3,6}|\d{1,3}(?:\.\d)?\s?[kK])([\s.,]*)(obo|o\.b\.o|firm|or best offer|negotiable|cash)\b", re.I)
RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
RE_URL = re.compile(r"(?:https?://|www\.)\S+|\b[\w-]+\.(?:com|net|org|us|biz|info)(?:/\S*)?\b", re.I)
SEP = r"[\s.\-\u2010-\u2015\u2212]"  # includes unicode hyphens/minus
RE_PHONE = re.compile(r"(?:\+?1" + SEP + r"?)?\(?(?<!\d)\d{3}\)?" + SEP + r"{0,3}\d{3}" + SEP + r"{0,3}\d{4}(?!\d)")
RE_ADDRESS = re.compile(
    r"\b\d{2,6}\s+(?:[NSEW]\.?\s+)?(?:[A-Za-z0-9]+\s+){1,4}"
    r"(?:st|street|ave|avenue|rd|blvd|boulevard|ln|lane|pkwy|parkway)\b\.?",
    re.I,
)


def price_forms(p):
    p = int(p)
    forms = {str(p), f"{p:,}", f"{p:,}".replace(",", ".")}
    if p % 100 == 0:
        forms |= {f"{p / 1000:g}k", f"{p / 1000:g}K"}
    return sorted(forms, key=len, reverse=True)


def mask(text, p):
    if not isinstance(text, str) or not text:
        return text
    t = RE_EMAIL.sub("[EMAIL]", text)
    t = RE_URL.sub("[URL]", t)
    t = RE_DOLLAR.sub("[PRICE]", t)
    t = RE_WORD_NUM.sub(lambda m: m.group(0)[: m.start(2) - m.start(0)] + "[PRICE]", t)
    t = RE_NUM_WORD.sub(lambda m: "[PRICE]" + m.group(2) + m.group(3), t)
    alts = "|".join(re.escape(f) for f in price_forms(p))
    t = re.sub(rf"(?<![\d,.])(?:{alts})(?![\d])", "[PRICE]", t)
    t = RE_PHONE.sub("[PHONE]", t)
    t = RE_ADDRESS.sub("[ADDRESS]", t)
    return t


df["description_masked"] = [mask(t, p) for t, p in zip(df.description, df.price)]

# ---------------------------------------------------------------- write files
MAIN_COLS = [
    "listing_id", "region", "state", "lat", "long",
    "manufacturer", "model", "year", "odometer", "condition", "cylinders", "fuel",
    "title_status", "transmission", "drive", "size", "type", "paint_color", "VIN",
    "seller_type", "posting_date",
]
train = df[df.split == "train"].sort_values("listing_id")
test = df[df.split == "test"].sort_values("listing_id")

train[MAIN_COLS + ["price_tier"]].to_csv(os.path.join(student_dir, "train.csv"), index=False)
test[MAIN_COLS].to_csv(os.path.join(student_dir, "test.csv"), index=False)
rng = np.random.default_rng(SEED)
pd.DataFrame({"ID": test.listing_id.values, "price_tier": rng.integers(1, 5, len(test))}).to_csv(
    os.path.join(student_dir, "sample_submission.csv"), index=False)
df.sort_values("listing_id")[["listing_id", "description_masked"]].rename(
    columns={"description_masked": "description"}).to_csv(os.path.join(student_dir, "descriptions.csv"), index=False)

test[["listing_id", "price_tier"]].rename(columns={"listing_id": "ID"}).to_csv(
    os.path.join(mentor_dir, "solution.csv"), index=False)
df.sort_values("listing_id")[
    ["listing_id", "split", "group_id", "price", "price_tier", "id", "url", "region_url", "image_url", "county"] + MAIN_COLS[1:]
].rename(columns={"id": "original_id"}).to_csv(os.path.join(mentor_dir, "full_with_price.csv"), index=False)

# ---------------------------------------------------------------- report
print("rows:", len(df), "| train:", len(train), "| test:", len(test), f"({len(test) / len(df):.1%})")
print("groups:", df.group_id.nunique(), "| largest group:", df.group_id.value_counts().max())
print("tier share train:", train.price_tier.value_counts(normalize=True).sort_index().round(3).to_dict())
print("tier share test: ", test.price_tier.value_counts(normalize=True).sort_index().round(3).to_dict())
print("rows in train that duplicate another train row (same group):", train.group_id.duplicated().sum())
m = df.description_masked.fillna("")
exact_left = np.mean([bool(re.search(rf"(?<![\d,.])(?:{'|'.join(map(re.escape, price_forms(p)))})(?![\d])", t)) for t, p in zip(m, df.price)])
print(f"exact price left in masked text: {exact_left:.3%}")
dollar_left = m.str.contains(r'[$]\s?[0-9]').mean()
print(f"'$'+digit left: {dollar_left:.3%}")
print(f"phone pattern left: {m.str.contains(RE_PHONE).mean():.3%} | email left: {m.str.contains(RE_EMAIL).mean():.3%}")
for tok in ["[PRICE]", "[PHONE]", "[EMAIL]", "[URL]", "[ADDRESS]"]:
    print(f"rows containing {tok}: {m.str.contains(re.escape(tok)).mean():.1%}")
