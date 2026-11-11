"""Step 1: filter the raw Craigslist listings to a sensible price range and bin price into 4 tiers.

Usage: python3 01_make_price_tiers.py <vehicles.csv> <vehicles_price_tier.csv>

Keeps listings priced $1,000-$150,000 (drops $0 placeholders and absurd prices), adds
`price_tier` (1-4, cut at $7,995 / $15,995 / $27,995) and drops `price`. The output is the
first input to 02_build_dataset.py, which re-attaches price by `id` and checks the tiers.
"""
import sys

import pandas as pd

TIER_BINS = [1000, 7995, 15995, 27995, 150000]

src, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(src, low_memory=False)
df = df[df.price.between(TIER_BINS[0], TIER_BINS[-1])].copy()
df["price_tier"] = pd.cut(df.price, TIER_BINS, labels=[1, 2, 3, 4], include_lowest=True).astype(int)
df.drop(columns="price").to_csv(out, index=False)

print(f"kept {len(df):,} rows")
print("tier share:", df.price_tier.value_counts(normalize=True).sort_index().round(3).to_dict())
