# OpsInsight: Data Preparation (EDA → Cleaning → Feature Engineering)

This folder takes your two raw zip files and turns them into **model-ready feature tables**, one notebook at a time.
Every notebook explains each step in plain English, and every code cell has already been run on your real data.
You can open the notebooks and see the expected output **before** you run anything yourself.

---

## What's inside

```
opsinsight/
├── README.md                ← this guide
├── requirements.txt         ← list of Python packages to install
├── .gitignore               ← stops big data files going into GitHub
├── notebooks/               ← run these IN ORDER
│   ├── 00_setup.ipynb                 create folders, extract the zips, check files
│   ├── 01_incidents_eda.ipynb         explore the incident data
│   ├── 02_incidents_cleaning.ipynb    clean it, one row per ticket
│   ├── 03_incidents_features.ipynb    build model features
│   ├── 04_sales_eda.ipynb             explore the 9 sales tables
│   ├── 05_sales_cleaning.ipynb        clean + join into one order table
│   ├── 06_sales_features.ipynb        build model features
│   ├── 07_baseline_check.ipynb        quick model to prove the features work and don't "cheat"
│   └── 08_analysis_json.ipynb         builds the full analysis JSON (the team contract)
├── contracts/               ← the JSON format everyone builds against
│   ├── CONTRACT.md                    meaning of every field
│   ├── example_incidents.json         real example from your data
│   └── example_sales.json
├── data/
│   ├── zips/        ← PUT YOUR TWO ZIP FILES HERE
│   ├── raw/         ← extracted CSVs (created by notebook 00, never edit)
│   ├── interim/     ← cleaned tables (created by 02 and 05)
│   └── processed/   ← final feature tables (created by 03 and 06)
└── reports/
    ├── figures/                         all charts as PNG (use them in slides)
    ├── *_quality_report.csv             what each cleaning step removed and why
    ├── *_feature_dictionary.csv         every feature, how it's made, proof it's known at prediction time
    └── baseline_results.csv             model scores
```

---

## Part 1: One-time setup on your computer (about 20 minutes)

### Step 1. Install Python
1. Go to **python.org/downloads** and download **Python 3.12**.
2. Run the installer.
   - **Windows:** on the first screen, tick **"Add python.exe to PATH"**, then click *Install Now*. This is the #1 beginner mistake, so don't skip it.
   - **Mac:** just click through the installer.
3. Check it worked. Open a terminal (Windows: press Start, type `cmd`, Enter. Mac: open *Terminal*) and type:
   ```
   python --version
   ```
   On Mac use `python3 --version`. You should see `Python 3.12.x`.

### Step 2. Install VS Code (the code editor)
1. Download from **code.visualstudio.com** and install.
2. Open VS Code → click the **Extensions** icon on the left (four squares) → search and install:
   - **Python** (by Microsoft)
   - **Jupyter** (by Microsoft)

### Step 3. Put the project in place
1. Unzip `opsinsight_data_prep.zip` somewhere easy, e.g. `Documents/opsinsight`.
2. Copy your **two dataset zip files** into `opsinsight/data/zips/`:
   - `...incidentmanagementprocessenrichedeventlog.zip`
   - `sales.zip`

   (Names don't matter. Notebook 00 looks inside each zip to decide where it goes.)

### Step 4. Open the project in VS Code
`File → Open Folder…` → choose the `opsinsight` folder (the one containing `requirements.txt`).

### Step 5. Create a virtual environment
A **virtual environment** ("venv") is a private box of Python packages just for this project, so it never clashes with other projects.

In VS Code open a terminal: menu **Terminal → New Terminal**. Then type:

**Windows**
```
python -m venv .venv
.venv\Scripts\activate
```
If you get a red error about "running scripts is disabled", run this once and try again:
```
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Mac / Linux**
```
python3 -m venv .venv
source .venv/bin/activate
```

You'll now see `(.venv)` at the start of the terminal line. That means it's active.

### Step 6. Install the packages
```
pip install -r requirements.txt
```
This takes a few minutes. It installs pandas (tables), matplotlib/seaborn (charts), scikit-learn + LightGBM (machine learning), pyarrow (fast file format) and Jupyter (notebooks).

---

## Part 2: Run the notebooks

1. In VS Code's left panel open `notebooks/00_setup.ipynb`.
2. Top right, click **Select Kernel** → **Python Environments** → choose the one with **`.venv`** in its name.
3. Click **Run All** (top of the notebook).
4. When it says *"All files ready"*, open the next notebook and repeat: **00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08**.

Order matters: 02 needs nothing but raw data, 03 needs 02's output, 06 needs 05's output, 07 needs 03 and 06.

| Notebook | Time to run | What you should see at the end |
|---|---|---|
| 00 | seconds | ✔ next to all 10 CSV files |
| 01 | ~30 s | charts + a summary table of 11 findings |
| 02 | ~30 s | "All checks passed ✔", 24,644 tickets, breach rate 36.6% |
| 03 | ~1 min | "No leakage columns ✔", 33 features |
| 04 | ~30 s | charts + a summary table of 12 findings |
| 05 | ~30 s | every join says "rows still 98,907 ✔", 96,019 model rows, late rate 8.1% |
| 06 | ~1–2 min | "No leakage columns ✔", 41 features |
| 07 | ~1 min | test AUC ≈ 0.77 (incidents) and ≈ 0.72 (sales) |
| 08 | ~1 min | two JSON files saved in `contracts/` |

**Running cell by cell instead of Run All:** click inside a cell and press `Shift + Enter`. Do this the first time. Read the explanation above each cell, run it, look at the output.

---

## Part 3: The method in one page (what you're actually doing)

**1. EDA (Exploratory Data Analysis):** look before you touch. What is one row? What's missing? What's odd? What do we predict and how often does it happen? We write every problem into a "to fix" table.

**2. Cleaning:** fix every problem from EDA, and record how many rows each fix removed (the *quality report*).
Key fixes in your data:
- Incidents: missing values hidden as `?`; dates in day/month/year; 141k *event* rows collapsed into 24.6k *ticket* rows; outcome columns kept separate (`final_…`).
- Sales: 9 tables summarised to one row per order **before** joining (otherwise orders get silently copied), geolocation reduced to one point per zip, impossible timelines removed.

**3. Feature engineering:** turn columns into signals a model can learn from, using **only information available at prediction time**:
- time (hour, weekday, promised days)
- severity / order contents
- **workload** (how busy the team or seller was right then)
- **history** (how this team / seller / route performed on *earlier, already finished* cases, smoothed toward the average)
- categories with rare values grouped into "Other"

**4. Check:** train a quick model on old data, test on newer data (**never a random split** for time-based data). Then deliberately add a "future" column to see how leakage inflates the score.

### The 4 traps these notebooks protect you from
| Trap | What goes wrong | Where it's handled |
|---|---|---|
| **Data leakage** | using info from the future (final reassignments, delivery date, review) → fake high score | 02, 03, 06, demo in 07 |
| **Duplicating rows in joins** | joining a many-rows table copies orders silently | `safe_merge` in 05 |
| **`groupby().first()`** | doesn't return the first row; mixes values from later rows | 02 uses `drop_duplicates` |
| **Hidden clock** | all-time counts only grow, so they secretly tell the model the date | 03 uses fixed windows instead |

---

## Part 4: Results so far (from the included run)

| | Incidents (SLA breach at opening) | Sales (late delivery at approval) |
|---|---|---|
| Rows | 24,644 tickets | 96,019 delivered orders |
| Target rate | 36.6% breach (22.5% in test period) | 8.1% late (5.3% in test period) |
| Features | 33 | 41 |
| Test AUC (honest) | **0.77** | **0.72** |
| Top-10% riskiest flagged | 52% breach vs 22.5% average | 10.3% late vs 5.3% average |
| Strongest signals | team, recent helpdesk throughput, team's recent breach rate, priority | customer state, promised days, season, seller dispatch history, distance |
| Root-cause headline | 22% breach with 0 reassignments → 96% with 8+ | late orders score 2.3★ vs 4.3★ on time |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not recognised (Windows) | Reinstall Python and tick "Add to PATH", then reopen VS Code |
| `ModuleNotFoundError: No module named 'pandas'` | The notebook isn't using `.venv`. Click the kernel name (top right) → pick `.venv` |
| `FileNotFoundError ... incident_event_log.csv` | Run notebook 00 first; check your zips are in `data/zips/` |
| `FileNotFoundError ... .parquet` | You skipped a notebook. Run them in order |
| Kernel crashes / "out of memory" | Close other apps; the geolocation file is 1M rows. 8 GB RAM is enough |
| Changed code and things look wrong | `Restart` the kernel (top bar) and **Run All** again |

## Using this with Git / GitHub
`.gitignore` already excludes `data/` and `.venv/` (datasets are big and shouldn't go into Git). Each teammate puts the zips in `data/zips/` and runs notebook 00.

## Data credits
UCI Incident Management Process Enriched Event Log (CC BY 4.0) · Olist Brazilian E-Commerce Public Dataset (CC BY-NC-SA 4.0).
