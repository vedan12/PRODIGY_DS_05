# PRODIGY_DS_05 — Traffic Accident Analysis (US Accidents)

**Prodigy InfoTech — Data Science Internship, Task 05**

## Task
Analyze traffic accident data to identify patterns related to road
conditions, weather, and time of day. Visualize accident hotspots and
contributing factors.

## Dataset
**US Accidents** (Moosavi et al.), a countrywide traffic accident dataset
(CC BY-NC-SA 4.0, research/educational use).

The full Kaggle file has ~7.7 million rows, which is too large to analyze in a
free notebook. This project uses `accidents_sample.csv`, a **10,000-row
sample** I obtained from a public GitHub repo
([vasilis6194/US-Accidents-Project](https://github.com/vasilis6194/US-Accidents-Project),
`3. Data Cleaning/cleaned_accidents.csv`). Per that repo's README, it is a
random sample of the Kaggle file `US_Accidents_March23.csv`, taken **after
removing every row with a missing value**.

> **Please read the limitations below.** That sampling method means this
> sample is *not* representative of the full dataset.

**Note on "Severity":** in this dataset, `Severity` (1–4) measures the
**impact on traffic** (1 = short delay, 4 = long delay). It is *not* an
injury or fatality scale. Below, "larger-impact" means severity 3–4.

### Citation
- Moosavi, S., et al. "A Countrywide Traffic Accident Dataset." 2019.
- Moosavi, S., et al. "Accident Risk Prediction based on Heterogeneous Sparse
  Data: New Dataset and Insights." ACM SIGSPATIAL, 2019.

## Data Cleaning
| Step | Result on this sample |
|---|---|
| Parse `Start_Time` (file mixes formats, with and without nanoseconds) | `format="mixed"`; 0 unparsable |
| Drop rows missing time / location / severity | 0 dropped |
| Drop duplicate accident IDs | 0 dropped |
| Blank or unusable weather → "Unknown" | 6 rows ("N/A Precipitation"); none were blank |
| Group the raw weather strings (53 distinct values) into 6 categories | my own grouping (e.g. "Light Freezing Rain" → Snow / Ice) |
| Feature engineering | hour, day of week, weekend flag, time-of-day bucket, severe flag (severity ≥ 3) |

No rows needed dropping **because the sample was pre-filtered for missing
values**. The script still handles missing data, so it works on the full
file.

## Charts (in `output/`)
| Chart | What it shows |
|---|---|
| `accidents_by_hour.png` | Share of accidents by hour, weekday vs. weekend |
| `day_hour_heatmap.png` | Day-of-week × hour heatmap |
| `weather_conditions.png` | Accidents by weather category |
| `road_features.png` | How often road features sit near an accident |
| `severity_by_road_feature.png` | Larger-impact share with vs. without a feature (95% CI) |
| `severity_by_time_of_day.png` | Larger-impact share by time of day (95% CI) |
| `hotspots_map.png` | Hexbin density map of accident locations |
| `top_states_cities.png` | Top 10 states and cities by accident count |

## Key Findings

**Time of day — the clearest pattern in the data**
- The busiest hour is **5pm (788 accidents)**; **29.6%** of all accidents
  happen between 3pm and 7pm, versus **11.5%** between midnight and 6am.
- Weekdays show two peaks: a smaller one around 7–8am and a larger one from
  about 3–5pm — a commute pattern. Weekends are flatter, peaking at **3pm**.
- Weekdays average **1,555** accidents per day vs. **1,114** per weekend day
  (about 40% more). Friday is highest (1,705), Sunday lowest (1,047).
- Weekends have relatively more late-night accidents: **12.6%** of weekend
  accidents happen between midnight and 4am vs. **5.7%** on weekdays.

**Weather**
- Clear/Fair 49.2%, Cloudy 36.8%, Rain/Storm 8.0%, Fog/Haze/Smoke 3.0%,
  Snow/Ice 2.9%.
- ⚠️ **This is not a risk ranking.** Most accidents occur in clear weather
  largely because clear weather is the most common weather. Judging whether
  rain or snow raises risk would need to know how often each condition
  occurs (exposure), which this dataset doesn't provide.

**Road features and impact**
- Features most often present near an accident: crossings (9.6%), traffic
  signals (9.1%), junctions (7.8%), stop signs (2.9%).
- **Junctions** are the one clear signal: **9.4%** of junction accidents were
  larger-impact (severity 3–4) vs. **4.9%** elsewhere (95% CI 7.5–11.6% vs.
  4.5–5.3%; n = 779 vs. 9,221). Traffic signals show a smaller gap (7.3% vs.
  5.0%). Crossings and stop signs show no clear difference.
- This is an association, not proof of cause. Junction accidents may simply
  block more traffic, and junctions are also more common in dense urban areas.

**Time of day and impact**
- Larger-impact shares by time bucket: night 5.2%, morning rush 5.7%, midday
  **4.1%**, afternoon rush 5.4%, evening **6.2%** (overall: 5.2%). Midday and
  evening differ, and their 95% intervals only just fail to overlap (4.9% vs.
  5.1%). But five groups are being compared and the sample is not
  representative, so I treat this as a weak hint, not a finding.

**Hotspots**
- Densest clusters appear around Los Angeles, Florida, Texas cities, Atlanta,
  and the Northeast corridor. By count, CA has 25.1% of accidents, FL 14.2%,
  VA 5.1%, TX 4.8%, NY 4.7%. Top cities: Miami (381), Los Angeles (231),
  Orlando (205), Dallas (143), Houston (132).
- Counts reflect population and traffic volume, not per-capita risk.

## Limitations
1. **The sample is not representative of the full dataset.** Evidence from
   the file itself: 100% of rows come from a single data source (`Source1`);
   94.0% have severity 2 and only 1.9% severity 3 (the full dataset is
   reported to be closer to ~80% severity 2 per other public analyses; I did
   not verify this against the full file); and 87.2% of rows are from
   2020–2022. The state and city rankings, and any overall rates, may not
   match the full dataset.
2. **Associations, not causes.** No finding here shows that a condition
   causes accidents.
3. **No exposure baseline** for weather or road features (see above).
4. **Small groups are noisy** — hence the confidence intervals and `n`
   labels on the severity charts.
5. **Weather categories are my own grouping** of the raw strings.
6. I have **not** run the script on the full 7.7M-row file; the
   `SAMPLE_FRAC` option is there for that, but is untested at full scale.

## How to run
```bash
pip install pandas numpy matplotlib seaborn
python task5_accident_analysis.py
```
To use the full Kaggle file, download `US_Accidents_March23.csv`, set
`DATA_PATH` to it and `SAMPLE_FRAC = 0.01` (or similar) at the top of the
script. A random sample of the full file would be a better basis for the
state, severity and overall-rate findings above.

## Files
- `task5_accident_analysis.py` — cleaning + analysis pipeline
- `accidents_sample.csv` — the 10,000-row sample used
- `PRODIGY_DS_05.ipynb` — Colab-ready notebook (downloads the sample itself)
- `output/` — the 8 charts above

---
*Part of the Data Science Internship @ Prodigy InfoTech (Oct 2026)*
#ProdigyInfoTech
