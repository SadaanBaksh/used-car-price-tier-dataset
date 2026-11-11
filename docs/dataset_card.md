# Used Car Price Tier Prediction

Given a set of features describing a used-car listing, predict its price tier.

## Overview

Used cars sell at very different prices, and what a car can fetch depends on many factors: how old it is, how far it has been driven, what kind of vehicle it is, what condition it is in, where it is being sold and who is selling it.

In this bootcamp you are given used-car listings posted on Craigslist across the United States. Each listing is described by its location, make and model, physical specifications, condition and title, and seller. Your task is to predict the price tier of each listing from these attributes.

## Files

| File | Description |
|---|---|
| `train.csv` | The training set, including the target `price_tier`. |
| `test.csv` | The test set. Same columns as `train.csv` without `price_tier`. |
| `sample_submission.csv` | A sample submission in the correct format. |
| `descriptions.csv` | **Optional.** The free-text description of every listing (train and test), joined on `listing_id`. |

## Target

`price_tier` is the listing's asking price grouped into 4 tiers, from 1 (cheapest) to 4 (most expensive).

## Columns

| Column | Description |
|---|---|
| `listing_id` | Unique identifier for the listing. |
| `region` | The Craigslist region (local market) the listing was posted in. |
| `state` | US state (two-letter code). |
| `lat`, `long` | Coordinates of the listing. |
| `manufacturer` | Vehicle make. |
| `model` | Vehicle model, as typed by the seller. |
| `year` | Model year. |
| `odometer` | Miles on the odometer, as entered by the seller. |
| `condition` | Seller-reported condition. |
| `cylinders` | Number of engine cylinders. |
| `fuel` | Fuel type. |
| `title_status` | Legal title status of the vehicle (clean, rebuilt, salvage, …). |
| `transmission` | Transmission type. |
| `drive` | Drive type (fwd, rwd, 4wd). |
| `size` | Size class of the vehicle. |
| `type` | Body type (sedan, SUV, pickup, …). |
| `paint_color` | Exterior color. |
| `VIN` | Vehicle Identification Number, when provided. |
| `seller_type` | Whether the listing was posted by a private `owner` or a `dealer`. |
| `posting_date` | Date and time the listing was posted. |
| `price_tier` | **Target** (train only). |

Columns are as entered by sellers, so expect missing and inconsistent values.

### `descriptions.csv` (optional)

| Column | Description |
|---|---|
| `listing_id` | Joins to `train.csv` / `test.csv`. |
| `description` | The listing's free-text description. Prices, phone numbers, emails, web links and street addresses have been replaced with the tokens `[PRICE]`, `[PHONE]`, `[EMAIL]`, `[URL]` and `[ADDRESS]`. |

This file is large (about 1 GB). Load only the columns or rows you need.

## Evaluation

Submissions are evaluated on **macro F1** across the 4 tiers.

## Submission format

For every listing in `test.csv`, predict one `price_tier`, using the listing's `listing_id` as the `ID`. The file must have a header and look like this:

```
ID,price_tier
0,2
3,1
5,4
etc.
```

## Source

Derived from the [Craigslist Cars & Trucks dataset](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) by Austin Reese on Kaggle (listings scraped April–May 2021).
