# Used Car Price Tier Dataset: Building a Leak-Free Teaching Dataset

**How I turned 426,880 raw Craigslist car listings into a clean, leak-free 4-class classification dataset for a machine learning bootcamp.**

**Project Mentor, MLP Project Bootcamp (Sep 2026 term) · IIT Madras BS Degree**

![Python](https://img.shields.io/badge/Python-3.12-blue) ![pandas](https://img.shields.io/badge/pandas-data%20wrangling-150458) ![scikit-learn](https://img.shields.io/badge/scikit--learn-StratifiedGroupKFold-orange) ![SciPy](https://img.shields.io/badge/SciPy-graph%20components-8caae6)

---

## Why I built it

The Sep 2026 MLP term project is a Kaggle competition: predict the **price tier (1–4)** of workspace rental listings, scored on **macro F1**. As a Project Mentor I needed a practice dataset for the bootcamp that:

- has **the same shape of problem**: 4 balanced price tiers, macro F1, a mix of location, physical, condition and seller features, high-cardinality columns, and a free-text field;
- is **different enough** that working on it doesn't solve the competition for anyone;
- is **honest**: no shortcut that lets a model read the answer, and no split that rewards memorising.

I built it from the public [Craigslist Cars & Trucks dataset](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) (listings scraped April–May 2021).

| Term project (workspaces) | Bootcamp dataset (cars) |
|---|---|
| metro / borough / zone, coordinates | state / region, lat / long |
| space class, access type | body type, size, drive |
| seats, workstations, zones | cylinders, odometer, year |
| compliance status | title status |
| operator class | seller type (owner / dealer) |
| repeated `operator_ref` → leakage risk | repeated VINs / reposts → leakage risk |
| facilities list | free-text descriptions |
| `price_tier`, 4 classes, macro F1 | `price_tier`, 4 classes, macro F1 |

---

## The result

| | |
|---|---|
| **Rows** | 380,369 listings |
| **Split** | 305,162 train / 75,207 test (19.8%), grouped and stratified |
| **Features** | 20 columns + optional masked descriptions (joined on `listing_id`) |
| **Target** | `price_tier`, 4 classes: 25.9 / 24.6 / 24.8 / 24.7% |
| **Metric** | Macro F1 |
| **Reproducible** | Two scripts, fixed seed (42) |

The column dictionary and the problem statement given to students are in [`docs/dataset_card.md`](docs/dataset_card.md).

---

## How it was built

```
vehicles.csv (426,880 rows)
   │  01_make_price_tiers.py
   │    keep $1,000–$150,000  →  bin into 4 tiers  →  drop price
   ▼
vehicles_price_tier.csv (380,369 rows)
   │  02_build_dataset.py
   │    derive seller_type from URL
   │    shuffle + new random IDs
   │    link reposts into groups (VIN / identical content) via graph connected components
   │    80/20 StratifiedGroupKFold split by group
   │    mask prices + personal info in descriptions
   ▼
student/  train.csv · test.csv · sample_submission.csv · descriptions.csv
mentor/   solution.csv · full_with_price.csv
```

### 1. Target: price → 4 tiers
I kept listings priced **$1,000–$150,000**. That removes the `$0` placeholders and the obviously fake prices (32,895 listings are priced at $0, and the maximum is about $3.7 billion). The remaining prices were cut at **$7,995 / $15,995 / $27,995**, roughly the quartiles, so the four tiers are almost perfectly balanced and the focus stays on features, not resampling.

### 2. A new feature from the URL: `seller_type`
Craigslist URLs contain `/cto/` (by owner) or `/ctd/` (by dealer). Pulling that out gave a clean, complete column (298,186 dealer, 82,183 owner) and one of the strongest signals: 54% of owner listings are tier 1. It mirrors the operator columns in the term project.

### 3. Remove anything that identifies rows
Price, raw description, original ID, URLs and image links were dropped from the student files, and `county` was dropped because it's 100% empty. Rows were **shuffled and given new IDs 0…N−1**, so the ID carries no region or time information.

### 4. Grouped split to stop repost leakage
The same car is often posted many times, across regions and by dealers (one VIN appears 235 times, and the largest repost group has 261 listings). With a random split, copies of one car land in both train and test, and the model gets rewarded for memorising.

To fix this I treated it as a **graph problem**: every listing is a node, linked to a key node for its VIN and a key node for a hash of (year, make, model, odometer, description). **Connected components** of that graph give one group per physical car or ad, even when the copies are chained through different keys. That produced **224,028 groups**. `StratifiedGroupKFold` then put each whole group on one side of the split, so **no car appears in both train and test**, while keeping the tier mix the same.

I deliberately left reposts *inside* train (125,943 rows repeat another train row). Students who validate with plain `KFold` see an inflated score, which leads straight into the lesson on `GroupKFold`. In the reference notebook, random K-fold reported **0.838** macro F1 against an honest held-out **0.712**.

### 5. Masking prices in the descriptions
About **42% of raw descriptions contain the exact asking price**. In my checks, reading the dollar amount from the raw text lifted macro F1 from ~0.81 to ~0.88: a model could simply read the answer. The masking replaces with `[PRICE]`:
- any `$` amount;
- the listing's own price in every common format (`15995`, `15,995`, `15.995`, `16k`);
- numbers next to price words ("asking 8500", "11,999 obo", "offer under 14,500").

To check it worked, I compared the numbers left in the text against the true price: they match **2.0%** of the time, against **2.2%** when prices are shuffled, i.e. no better than chance.

### 6. Privacy masking
Phone numbers, emails, web links and street addresses are replaced with `[PHONE]`, `[EMAIL]`, `[URL]` and `[ADDRESS]`, using regexes that handle unicode dashes and common formatting. It's regex-based, so a small fraction of unusually formatted numbers and addresses can slip through. That's one reason the generated files aren't published here.

### 7. Messy values left in on purpose
Cleaning is part of the exercise, so realistic data errors stay in:

| Issue | Count (train + test) |
|---|---|
| `year` before 1990 (min 1900) | 11,694 |
| `odometer` < 100 (placeholder `1` alone: 1,714) | 4,327 |
| `odometer` > 500,000 (max 10,000,000) | 1,111 |
| Coordinates outside the US | 98 |
| `model` free text: 27,201 spellings, 13,961 appear once | |
| Missing: `size` 72%, `cylinders` 41%, `condition` 38%, `VIN` 38% | |

Missingness itself carries signal (e.g. `VIN` is missing far more often in tier 1), which makes a good case for missing-indicator features over plain imputation.

---

## What I learnt

- **Leakage hides in places you don't look.** The free-text field contained the target in 42% of rows. Before masking, a text model could score ~0.88 without learning anything about cars.
- **Measure that a fix worked.** Comparing leftover numbers with true vs. shuffled prices turned "I think the masking is good" into a number.
- **Duplicates need a graph, not a `drop_duplicates`.** Reposts are linked through different keys (same VIN in one pair, same text in another), so connected components were needed to group them.
- **The split defines what a score means.** A grouped, stratified split makes the leaderboard reward generalisation instead of memorisation.
- **Teaching data needs designed difficulty.** Balanced classes, realistic noise and one well-known trap (reposts) make for useful lessons.

---

## Repository structure

```
├── README.md
├── docs/
│   └── dataset_card.md          # Problem statement, column dictionary and metric given to students
├── scripts/
│   ├── 01_make_price_tiers.py   # Filter price range, bin into 4 tiers
│   └── 02_build_dataset.py      # seller_type, new IDs, repost groups, grouped split, masking, output files
└── requirements.txt
```

## Reproducing it

The generated files aren't included: the descriptions may still contain a little personal information, and the test labels are kept private. To rebuild everything from the public source:

1. Download `vehicles.csv` from [Kaggle](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data).
2. Run:

```bash
pip install -r requirements.txt
python3 scripts/01_make_price_tiers.py vehicles.csv vehicles_price_tier.csv
python3 scripts/02_build_dataset.py vehicles_price_tier.csv vehicles.csv output/
```

The second script prints a report: row counts, group sizes, tier shares, and how much price, phone and email text is left after masking.
