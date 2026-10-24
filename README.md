# OpsInsight

Upload your support tickets or sales orders. OpsInsight tells you which items are at risk and why, what to fix first, how much each fix is worth, and afterwards whether it worked.

Built for Eternal's hackathon problem DEV_103 by Arpita (data), Abhishek (risk model), Roshan (backend and AI engine) and Divesh (forecast and dashboard).

## What it does

| Step | What you get |
| --- | --- |
| Upload | CSV, Excel or a database table; recognised automatically, columns matched for you |
| Analyse | KPIs, trends, root causes, unusual weeks |
| Predict | risk for every new ticket or order, with its top 3 reasons |
| Forecast | next week's load per team or region, and the overloaded days |
| Decide | findings ranked by size × confidence × urgency, a workload plan, actions with expected impact |
| Approve | approve, reject or edit each action; every decision is kept |
| Check | did the approved action work? |
| Ask | questions in plain words, answered only from your data's facts |

Tested on two public datasets: the UCI Incident Management event log (141,712 rows, 24,918 tickets) and the Olist Brazilian e-commerce dataset (9 files, about 99,000 orders).

## Run it (about 10 minutes)

You need Python 3.11+ and Node.js 18+.

Windows (PowerShell):

```powershell
git clone https://github.com/EC-Arpita/OpsInsight.git
cd OpsInsight
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd api
uvicorn opsinsight.main:app --port 8000
```

Mac / Linux:

```bash
git clone https://github.com/EC-Arpita/OpsInsight.git
cd OpsInsight
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd api
uvicorn opsinsight.main:app --port 8000
```

In a second terminal:

```bash
cd OpsInsight/frontend
npm install
npm run dev
```

Open http://localhost:5173. The two demo analyses load at once. To try your own data, click **Upload data**. The API's test page is at http://localhost:8000/docs.

No API running? The dashboard still shows the demo analyses from `frontend/public`.

## Project layout

| Folder | What | Owner |
| --- | --- | --- |
| `api/opsinsight/prep` | cleaning, features, analysis | Arpita |
| `api/opsinsight/models` | risk model and its reasons | Abhishek |
| `api/opsinsight/forecast` | next week's workload | Divesh |
| `api/opsinsight/engine` | uploads, decisions, simulation, approvals, outcomes, Ask | Roshan |
| `api/opsinsight/main.py` | the API | Roshan |
| `frontend` | the dashboard | Divesh |
| `contracts` | the analysis JSON format everyone builds against | all |
| `notebooks` | the data work step by step (see DATA_PREP.md) | Arpita |

## Tests

```bash
cd api
python -m pytest tests -v
```

21 tests: upload checks, decisions, outcomes, Ask, the risk model, the forecast and file recognition.

## Rebuild the example analyses (needs the data files)

Put the two dataset zips in `data/zips`, run notebooks 00 to 08 (DATA_PREP.md), then from `api`:

```bash
python -m opsinsight.forecast.run
python -m opsinsight.models.refresh
python -m opsinsight.engine.refresh
```

## Honest limits

- Impact estimates rest on stated assumptions (for example, fast-tracking prevents 30% of breaches),
  shown on every card with a range, until a pilot measures them.
- The risk model is trained on the UCI and Olist data; retraining per company is the next step.
- Prototype for one machine: no login yet, so do not upload real company data to a shared server.
- "Ask" shows facts directly; with Ollama installed it writes answers and every number is checked.
