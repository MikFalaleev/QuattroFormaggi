# raw_mini — test fixture for step 3

Hand-sized copy of the raw tables in the style of `yogape/logistics-operations`: 10 loads,
3 routes, 3 customers, 20 delivery events (one Pickup and one Delivery per load). Same column
sets and value formats as the real CSVs; all values are consistent (routes match events,
pickup date equals load_date, transit 0–3 days). Expectations for it: `tests/test_profile.py`.
