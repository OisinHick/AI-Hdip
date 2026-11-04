Lecture 6: run the data pipeline demo (in Google Colab or on your own computer)

Data Handling and Infrastructure for AI | MSc

In the lecture you watched a data pipeline being built with Apache Beam. In this notebook you run it, step by step, in Google Colab (free, in your browser, nothing to install) or on your own computer. You need no Google Cloud project, no credits, and it costs nothing.

By the end you will have:

    read the pipeline’s code, piece by piece, with an explanation of each piece;
    run its automatic tests and watched them pass;
    run the pipeline on a small made-up sample, and read its counters and its quarantine (the rows it rejected);
    run it on real Dublin Bikes data (about a million rows) and checked its answer against a second, independent method;
    done two exercises of your own.

You do not have to understand everything on the first read. Every step tells you what it does, what you should see, and what to do if it does not work. If you get stuck for more than a few minutes, ask your tutor.
Time plan (about 90 to 120 minutes)
Part 	What you do 	Minutes
0 	Set up and check (in Colab, run the Part 0a cell first) 	10
1 	Read the pipeline’s code 	15
2 	Run the automatic tests 	5
3 	Make and look at the sample data 	10
4 	Run the pipeline on the sample 	10
5 	Exercise 1: break it on purpose 	15
6 	Run it on the real data 	15
7 	Exercise 2: investigate the real results 	15
8 	Wrap-up 	5

If you are short of time, do Parts 0 to 5 first. They use only the small sample and take no downloads.
How this notebook works

    A notebook is made of cells. Text cells explain. Code cells contain Python that you run.
    Run a cell by clicking in it and pressing Shift + Enter. Run the cells in order, from top to bottom. Each cell uses results from the cells above it.
    A cell that is running shows [*] next to it. Wait for it to finish before you run the next one.
    Blanks. Some code cells contain ___ (three underscores). Replace each one with your answer. Until you do, you will see a red NameError: that is expected, not a mistake.
    Checks. After most tasks a check cell prints PASS (your answer is right) or CHECK AGAIN (look at your answer again). A pass shows the number is right, not that you understand why: read the explanations too.
    Written answers. Cells headed Your answer are for one or two sentences. Double-click the cell, type, and press Shift + Enter. Nothing is handed in.
    Hints are hidden under Hint. Click to open one. Try for two minutes first.

If you need to start again (for example, after an error you cannot fix): restart the kernel, then run the cells again from the top. Do not use “Run all”: it stops at the first blank. To get back to where you were, run everything above the cell you are on.
	Google Colab 	Jupyter
Restart 	Runtime > Restart session 	Kernel > Restart Kernel
Run everything above 	Runtime > Run before (Ctrl + F8) 	Run > Run All Above Selected Cell
Stop a cell that runs forever 	Runtime > Interrupt execution 	Kernel > Interrupt Kernel

Using Colab? Two things to know. (1) Colab gives you a temporary computer: if you are idle for a long time or close the tab, it is reset and everything you created is lost (including the downloaded data). Just run the cells again from Part 0a. (2) Your answers are only kept if you save the notebook: File > Save a copy in Drive, or File > Download > .ipynb.
Part 0a. Google Colab setup (skip this if you are on your own computer)

Colab starts with an empty computer that does not have the demo code or Apache Beam. The cell below fixes both: it downloads the code from GitHub and installs Apache Beam (about a minute). Run it once, at the start of every Colab session. On your own computer it does nothing.

What the cell does, step by step:

    It checks whether it is running in Colab.
    It downloads the repository into /content/lecture6_demo (Colab’s working folder) with git clone.
    It installs Apache Beam (and pytest, the testing tool) with pip. It installs the core Beam library only: the Google Cloud extras are not needed for this notebook, and they can clash with Colab’s own libraries.

If Colab then shows a message saying you must restart the runtime (this can happen when an installed library replaces one that was already loaded): choose Runtime > Restart session, then run this cell again (it will skip what is already done) and carry on with Part 0.

# --- Part 0a: Google Colab setup (safe to run anywhere: it does nothing outside Colab) ---------
import sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/AbNeil/lecture6_demo"

def pip_install(*packages):
    return subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], capture_output=True, text=True)

if not IN_COLAB:
    print("You are not in Colab: nothing to do here. Go on to Part 0.")
else:
    repo = Path("/content/lecture6_demo")
    if repo.exists():
        print("The code is already downloaded:", repo)
    else:
        print("Downloading the code from GitHub ...")
        result = subprocess.run(["git", "clone", "-q", REPO_URL, str(repo)], capture_output=True, text=True)
        print(result.stdout + result.stderr if result.returncode else "Done.")
    try:
        import apache_beam
        print("Apache Beam is already installed, version", apache_beam.__version__)
    except ImportError:
        print("Installing Apache Beam (about a minute) ...")
        result = pip_install("apache-beam==2.77.0", "pytest")
        if result.returncode:
            print("That exact version did not install; trying the newest version instead ...")
            result = pip_install("apache-beam", "pytest")
        print((result.stdout[-500:] + result.stderr[-1500:]) if result.returncode else "Installed.")
    print("\nIf Colab now asks you to RESTART the runtime: Runtime > Restart session, then run this cell again.")

You should see “Downloading the code from GitHub … Done.” and “Installed.” (or “already installed” on a second run).
If it fails
	
	
	
	
Part 0. Check your setup (10 min)

Before anything else we check that the notebook can find the demo code and the libraries it needs. (In Colab, run the Part 0a cell above first.) Run the cell below (Shift + Enter). It does four things: loads some helper tools, finds the folder called lecture6_demo that holds the pipeline, switches into it, and prints the versions of the libraries.

What is the “demo folder”? It is the folder from the GitHub repository that contains bike_pipeline.py (the pipeline), test_pipeline.py (its tests), and the scripts. This notebook lives in a sibling folder, so it looks for ../lecture6_demo.

# --- Part 0: helper tools and setup check (just run this cell) ---------------------
import os, sys, re, json, glob, time, inspect, shutil, subprocess, urllib.request
from pathlib import Path
import pandas as pd

PY = sys.executable          # the Python that runs this notebook: we use it for every command

def check(label, ok):
    """Self-check: prints PASS or CHECK AGAIN."""
    print(("PASS         " if bool(ok) else "CHECK AGAIN  ") + label)

def find_demo():
    """Look for the folder that contains bike_pipeline.py."""
    here = Path.cwd().resolve()
    for candidate in [here / ".." / "lecture6_demo", here / "lecture6_demo", here / "lecture6_demo" / "lecture6_demo",
                      Path("/content/lecture6_demo/lecture6_demo"), here]:
        if (candidate.resolve() / "bike_pipeline.py").exists():
            return candidate.resolve()
    return None

DEMO = find_demo()
if DEMO is None:
    raise SystemExit("I cannot find bike_pipeline.py. In Google Colab: run the Part 0a cell first. On your own computer: "
                     "is this notebook inside the lecture6_notebooks folder, next to the lecture6_demo folder? See the README, 'Troubleshooting'.")
os.chdir(DEMO)
sys.path.insert(0, str(DEMO))

def run(args, show=True):
    """Run a command (a list of words) in the demo folder, print its output as it appears,
    and return (exit code, all the output)."""
    process = subprocess.Popen(args, cwd=DEMO, env=dict(os.environ, PYTHONUNBUFFERED="1"),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in process.stdout:
        lines.append(line)
        if show:
            print(line, end="")
    process.wait()
    return process.returncode, "".join(lines)

import logging
logging.getLogger().setLevel(logging.ERROR)      # hides a harmless warning about Google Cloud libraries that we do not use
import apache_beam, matplotlib
logging.getLogger().setLevel(logging.WARNING)
print("Demo folder :", DEMO)
print("Python      :", sys.version.split()[0], "  (", PY, ")")
print("Apache Beam :", apache_beam.__version__)
print("pandas      :", pd.__version__)
print("matplotlib  :", matplotlib.__version__)

You should see the demo folder path (ending in lecture6_demo) and four version numbers, with no red error. (A pink or grey line starting WARNING:root:apitools not found is harmless: it only says that Google’s cloud libraries are not installed, and we do not use them here. Ignore it.)
If it fails
	
	
	
	
Part 1. Read the pipeline’s code (15 min)

A data pipeline is an automated, repeatable series of steps that moves data from where it is produced to where it is used, checking and transforming it on the way. Ours does five things:

CSV files ──► read each file ──► check every row ──┬─ good rows ──► hourly windows ──► averages per station ──► add station name ──► results
                                                    └─ bad rows  ──► quarantine (with the reason) ───────────────────────────────────► rejected rows

Its question is simple: for each bike station and each hour, how many bikes were available on average, at least, and at most?

The code is in bike_pipeline.py. We load it into this notebook and print its parts, with an explanation of each. Read the code, then the explanation next to it. You do not need to memorise any of it.

# --- Part 1: load the pipeline and print its pieces -------------------------------
import bike_pipeline as bp

def show(thing):
    """Print the source code of a function or class from bike_pipeline.py."""
    print(inspect.getsource(thing))

print("The three columns every row MUST have:", bp.REQUIRED)
print("The name of the quarantine output     :", repr(bp.BAD))

Piece 1: reading one file (ReadCsvFile)

Run the cell, then read the notes under it.

show(bp.ReadCsvFile)

What to notice

    A DoFn is a class with a process method. For each input it may produce any number of outputs (here: one per row).
    The input is a whole file, not a row. That lets us read that file’s own header with csv.DictReader. Each row becomes a dictionary like {"station_id": "42", "num_bikes_available": "5", ...}, so we find columns by name, never by position. This matters because files can have different columns.
    utf-8-sig quietly removes an invisible marker that some CSV files start with.
    row["_source"] = name remembers which file each row came from, so a problem can be traced back.
    The counters (files_read, rows_read, empty_files) are how the pipeline reports what happened. An empty file gives zero rows and is counted, not crashed on.

Piece 2: reading times and numbers safely

show(bp.RowError)
show(bp.parse_timestamp)
show(bp.to_int)

What to notice

    RowError is our own kind of error. Its message is the reason a row is rejected.
    parse_timestamp accepts either a number of seconds since 1970 or text such as 2024-06-01 00:05:00. If the text has no time zone (as in the real files), we assume UTC. That is an assumption about the data, and a good pipeline says so.
    to_int accepts 12 and 12.0 but turns anything else (such as "many") into a RowError with a reason that names the column.

Piece 3: the data contract as code (ValidateRow)

show(bp.ValidateRow)

What to notice

    This is the data contract from Lecture 5, now running on every row. In order, it checks: the three required columns are present and not empty; the station and the bike count are whole numbers; the time can be read; the bike count is not negative.
    A row that breaks a rule is not dropped silently and does not crash the job. It goes to a second output, named BAD, together with the reason, the file name and the raw row. That is the quarantine.
    over_capacity is a plausibility rule: more bikes than the station’s capacity is odd but possible, so we keep the row and just count it (Lecture 5: flag, do not delete).

Piece 4: how the averages are computed (BikeStats)

show(bp.BikeStats)

What to notice

    A CombineFn computes an aggregate in four small steps so that many machines can share the work: start empty (create_accumulator), add one reading (add_input), join partial results from different machines (merge_accumulators), and produce the answer (extract_output).
    We carry the total and the count, not the average itself, because two averages cannot be merged correctly without knowing how many readings each came from.

Piece 5: the whole pipeline in one place (build)

This is the most important piece. Read it top to bottom: every line is one step, written | "Name" >> step.

show(bp.build)

What to notice

    stations is a small side branch: it reads the station reference file (number and name).
    checked finds the files (MatchFiles), opens them (ReadMatches), reads their rows (ReadRows) and validates them (Validate), with two outputs: good and the BAD quarantine.
    hourly continues with the good rows: Stamp (give each reading its real time), Hourly (cut time into windows), Stats (average, minimum, maximum per station and window), and Enrich (add the station name from the side input).

Three small tasks: prove you read it

Answer by reading the code printed above, not by guessing.

required_columns = ___        # the three columns every row must have (see bp.REQUIRED above)
window_seconds   = ___        # how many seconds long is one window? (look in build(): FixedWindows(...))
bad_output_name  = ___        # the name of the quarantine output (a word in quotes)

Hint

check("the three required columns", set(required_columns) == set(bp.REQUIRED))
check("window length is one hour (3600 seconds)", window_seconds == 3600)
check("quarantine output name", bad_output_name == bp.BAD)

Part 2. Run the automatic tests (5 min)

A pipeline is code, so we test its pieces with small inputs whose right answer we know. The file test_pipeline.py has 5 tests. They check, for example, that bad rows are sent to the quarantine with the right reasons, and that the averages are computed correctly.

Run the cell below. It runs pytest, the standard Python testing tool, in the demo folder. It takes 10 to 30 seconds.

code, out = run([PY, "-m", "pytest", "-q", "test_pipeline.py"])
print()
check("all tests passed (exit code 0)", code == 0)

You should see a line ending 5 passed (the number of warnings and the time can differ).

Why do tests matter for pipelines? A change that silently breaks the validation could let bad data into the results without any error message. Tests catch that before the pipeline runs on real data. In the cloud, where every run costs credits, they also stop you paying to run a bug.
If it fails

Part 3. Make and look at the sample data (10 min)

Before running any pipeline, look at the data. That is the first rule of data engineering.

The script make_sample_data.py creates a tiny made-up sample (the numbers are invented, but the station names are real). It contains three files and a few deliberately bad rows, so that we can watch the quarantine work:

    ..._sample_2024.csv: 15 columns, with 4 deliberately bad rows;
    ..._sample_2026.csv: 9 columns (a different layout), plus two odd rows: a station that is not in the station list, and a row with more bikes than the capacity;
    ..._sample_empty.csv: only a header (an empty file);
    stations.csv: the station reference list (number and name).

code, out = run([PY, "make_sample_data.py"])
raw_files = sorted(Path("sample/raw").glob("*.csv"))
print()
for f in raw_files:
    print(f"{f.name:55s} {f.stat().st_size:>7} bytes")

Now look inside the files. First the raw first lines (exactly as they are on disk), then as a table.

for f in raw_files:
    print("---", f.name)
    with open(f, encoding="utf-8") as handle:
        for _ in range(3):
            print(handle.readline().rstrip())
print()
print("The 2024 file as a table:")
pd.read_csv(raw_files[0]).head(4)

Two things to notice: the 2024 and 2026 files have different columns (15 against 9), and the third file has a header but no data rows. A pipeline has to cope with both.
Task: count the rows before the pipeline does

wc -l style: count the lines of each file, then work out how many data rows there are (every file has one header line).

line_counts = {f.name: ___ for f in raw_files}     # the number of lines in each file
total_data_rows = ___                                # lines in total, minus one header line per file
print(line_counts)
print("data rows:", total_data_rows)

Hint

check("189 lines in total", sum(line_counts.values()) == 189)
check("186 data rows (lines minus 3 headers)", total_data_rows == 186)

Part 4. Run the pipeline on the sample (10 min)

Now the main event. We run the pipeline right here, on this notebook’s own computer (your laptop, or the free Colab machine), using Beam’s DirectRunner (the runner that works locally, with no cloud pipeline service). The command has four options. Read them before you run:
Option 	Meaning
--runner DirectRunner 	run here, on this computer (no cloud pipeline service)
--input "sample/raw/*.csv" 	which files to read: every CSV in that folder
--stations sample/ref/stations.csv 	the small reference file with station names
--output out/sample/hourly 	where to write the results (a file name prefix)

The cell below wraps the command in a small function, run_pipeline, because we will use it again. It also reads the counters printed at the end, and turns them into a Python dictionary.

def run_pipeline(input_glob, stations, output_prefix, extra=()):
    """Run bike_pipeline.py locally. Returns (exit code, counters as a dict, seconds taken)."""
    args = [PY, "bike_pipeline.py", "--runner", "DirectRunner", "--input", input_glob,
            "--stations", stations, "--output", output_prefix, *extra]
    start = time.time()
    code, out = run(args)
    counters = {m.group(1): int(m.group(2)) for m in re.finditer(r"^\s{2}(\w+)\s+(\d+)\s*$", out, flags=re.M)}
    return code, counters, time.time() - start

def read_jsonl(pattern):
    """Read all the result files that match a pattern into one table."""
    frames = [pd.read_json(f, lines=True) for f in sorted(glob.glob(pattern)) if os.path.getsize(f) > 0]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

shutil.rmtree("out/sample", ignore_errors=True)
code, counters, seconds = run_pipeline("sample/raw/*.csv", "sample/ref/stations.csv", "out/sample/hourly")
print(f"\nExit code {code}; took {seconds:.0f} seconds")

You should see (after a few INFO lines, and possibly a harmless WARNING:root:apitools not found line, which you can ignore) a block like this, and an exit code of 0:

Counters:
  bad_rows        4
  empty_files     1
  files_read      3
  good_rows       182
  over_capacity   1
  rows_read       186
  unknown_station 1

The counters are the pipeline’s test report. Read them like one:
Counter 	What it tells you
files_read 3, empty_files 1 	three files were read, and one of them had no rows
rows_read 186 	rows read in all (the same 186 you counted in Part 3)
good_rows 182 and bad_rows 4 	rows that passed, and rows quarantined
over_capacity 1 	a row with more bikes than capacity: kept but counted
unknown_station 1 	one hourly result whose station is not in the station list: kept but counted

Now load the two results into tables.

hourly = read_jsonl("out/sample/hourly-0*.jsonl")        # the results: one row per station per hour
bad    = read_jsonl("out/sample/hourly_bad-0*.jsonl")     # the quarantine: rows that broke a rule
print("Hourly results:", hourly.shape, "(rows, columns)")
display(hourly.head(5))
print("Quarantine:", bad.shape)
display(bad[["reason", "source"]])

Tasks: read the results with pandas

stations_in_output = ___      # how many different stations appear in the hourly results?
unnamed_rows       = ___      # how many hourly rows have NO station name (station_name is missing)?
print(stations_in_output, unnamed_rows)

Hint

check("every row is accounted for: rows_read = good_rows + bad_rows", counters["rows_read"] == counters["good_rows"] + counters.get("bad_rows", 0))
check("6 stations appear (5 real ones and the unknown station 999)", stations_in_output == 6)
check("the unnamed rows match the unknown_station counter", unnamed_rows == counters.get("unknown_station", 0))
check("the quarantine has exactly 4 rows", len(bad) == 4)

Your answer (1 or 2 sentences, double-click to edit). rows_read is 186, good_rows is 182 and bad_rows is 4. Why is it reassuring that the three numbers add up?

Write your answer here (double-click to edit, then press Shift + Enter).

Your answer (1 or 2 sentences). The 4 bad rows were not dropped and did not crash the pipeline. Why is that better than either of those two options?

Write your answer here (double-click to edit, then press Shift + Enter).
Part 5. Exercise 1: break it on purpose (15 min)

The best way to understand a data contract is to try to break it. In this exercise you write a small CSV file with deliberately bad rows. You predict what the pipeline will do, then run it and see whether you were right.
Step 1: write your test file

The six good rows are written for you. Your job is to fill in the seven rows marked ___. Each one tests one rule. Use only these kinds of value:
Row 	What to put in the blank 	The rule it tests
7 	a negative number of bikes 	bikes must not be negative
8 	an empty string "" as the station id 	the station id is required
9 	some text that is not a time, such as "yesterday" 	the time must be readable
10 	a word instead of a number of bikes, such as "many" 	the bike count must be a number
11 	more bikes than the capacity (the capacity is 20) 	odd but valid: kept and counted

Rows 12 (a station that is not in the station list) is written for you. Station 999 is not in the list.

stations_file = "sample/ref/stations.csv"      # the same station list we used before
os.makedirs("exercise1/raw", exist_ok=True)

def row(station, time, bikes, capacity=20):
    return {"station_id": station, "last_reported": time, "num_bikes_available": bikes, "capacity": capacity}

rows = [
    # six good rows (given)
    row(42, "2024-06-01 08:00:00", 5),
    row(42, "2024-06-01 08:20:00", 7),
    row(30, "2024-06-01 08:05:00", 12),
    row(30, "2024-06-01 08:25:00", 0),      # zero bikes is a VALID reading: an empty station
    row(54, "2024-06-01 09:10:00", 3),
    row(54, "2024-06-01 09:30:00", 9),
    # YOUR rows: one of each kind of bad (or odd) row
    row(42, "2024-06-01 08:40:00", ___),            # row 7: a negative number of bikes
    row(___, "2024-06-01 08:40:00", 5),             # row 8: an empty station id
    row(42, ___, 5),                                # row 9: a time that cannot be read
    row(42, "2024-06-01 08:40:00", ___),            # row 10: a word instead of a number
    row(42, "2024-06-01 08:50:00", ___),            # row 11: more bikes than the capacity of 20
    # given: a station that is not in the station list
    row(999, "2024-06-01 08:00:00", 4),
]
pd.DataFrame(rows).to_csv("exercise1/raw/my_bikes.csv", index=False)
print(pd.DataFrame(rows))

Hint

Step 2: predict before you run

Count your rows. Which ones are good, which are bad, which are odd-but-valid? Fill in five numbers below, without running anything yet. Think about:

    rows_read: all the data rows in your file;
    bad_rows: the rows that break a rule (the first four of your special rows);
    good_rows: everything else, including the odd-but-valid rows;
    over_capacity: how many rows have more bikes than capacity;
    unknown_station: how many hourly results belong to a station that is not in the list. (Careful: it counts results (one per station per hour), not rows.)

prediction = {
    "rows_read": ___,
    "good_rows": ___,
    "bad_rows": ___,
    "over_capacity": ___,
    "unknown_station": ___,
}
prediction

Hint

Step 3: run the pipeline and compare

shutil.rmtree("exercise1/out", ignore_errors=True)
code, actual, seconds = run_pipeline("exercise1/raw/*.csv", stations_file, "exercise1/out/hourly")
print("\nYour prediction:", prediction)
print("Actual counters:", {k: actual.get(k, 0) for k in prediction})

for name in prediction:
    check(f"{name}: predicted {prediction[name]}, actual {actual.get(name, 0)}", prediction[name] == actual.get(name, 0))

Now look at the quarantine. Which reason did each of your bad rows get?

my_bad = read_jsonl("exercise1/out/hourly_bad-0*.jsonl")
display(my_bad[["reason", "source"]])
expected_reasons = {"negative_bikes", "missing:station_id", "bad_timestamp", "bad_number:num_bikes_available"}
check("your four bad rows got four different, correct reasons", set(my_bad["reason"]) == expected_reasons)

Your answer (2 or 3 sentences). Did every prediction come true? If one was wrong, which one, and what did you misunderstand? Which of your rows were odd but kept, and why does the pipeline keep them instead of rejecting them?

Write your answer here (double-click to edit, then press Shift + Enter).
Part 6. Run it on the real data (15 min)

Now the real thing. Dublin City Council publishes real Dublin Bikes data (licence CC BY 4.0, https://data.smartdublin.ie/dataset/dublinbikes-api). We use two monthly files: June 2024 and February 2026. Together they are about 100 to 130 MB and about a million rows.

Think about your connection first. The download is about 130 MB (in Colab it is fast, and the files are kept on Colab’s temporary disk, so they disappear when the session is reset). If you are on a slow connection, ask your tutor: they may have the files on a memory stick or a shared folder. If you want to skip Parts 6 and 7 for now, set RUN_REAL = False in the next cell. Every later real-data cell will then skip itself.

RUN_REAL = True        # set to False to skip all the real-data steps (Parts 6 and 7)
print("Real-data steps are", "ON" if RUN_REAL else "OFF (skipped)")

Step 1: download the files

The next cell downloads the three files into data/raw and data/ref (inside the demo folder). If a file is already there it is not downloaded again. It prints a message as it goes.
If the download fails
	
	
	
	

BASE = "https://data.smartdublin.ie/dataset/33ec9fe2-4957-4e9a-ab55-c5e917c7a9ab/resource"
FILES = {
    "data/raw/dublin-bikes_station_status_062024.csv":
        os.environ.get("JUN2024_URL", BASE + "/022119a2-7679-4742-9b0a-c39dce0f38c1/download/dublin-bikes_station_status_062024.csv"),
    "data/raw/dublin-bikes_station_status_022026.csv":
        os.environ.get("FEB2026_URL", BASE + "/91098ee8-7ea7-4c70-b96d-face618a02b6/download/dublin-bikes_station_status_022026.csv"),
    "data/ref/dublin.csv":
        os.environ.get("STATIONS_URL", BASE + "/2dec86ed-76ed-47a3-ae28-646db5c5b965/download/dublin.csv"),
}

def download(url, path):
    """Download one file in pieces, printing the progress."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (Lecture 6 demo)")]
    with opener.open(url, timeout=60) as response, open(path, "wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        done, next_mark = 0, 0.1
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
            done += len(block)
            if total and done / total >= next_mark:
                print(f"   {path.name}: {done / total:.0%}")
                next_mark += 0.1

if RUN_REAL:
    try:
        for path, url in FILES.items():
            if Path(path).exists() and Path(path).stat().st_size > 0:
                print("already here:", path)
            else:
                print("downloading :", path)
                download(url, path)
        print()
        for path in FILES:
            print(f"{Path(path).stat().st_size / 1e6:8.1f} MB  {path}")
    except Exception as error:
        print("DOWNLOAD PROBLEM:", error)
        print("See the table above: run the download with curl instead, or ask your tutor.")

Step 2: look at the real data first

Print the header (the first line) and the first row of each file, then count the rows. This is where you meet the real columns and the real timestamp format.

if RUN_REAL:
    real_files = sorted(Path("data/raw").glob("*.csv"))
    for f in real_files:
        print("---", f.name)
        with open(f, encoding="utf-8") as handle:
            print(handle.readline().rstrip())
            print(handle.readline().rstrip())
    print()
    real_lines = {f.name: sum(1 for _ in open(f, encoding="utf-8")) for f in real_files}
    for name, n in real_lines.items():
        print(f"{name}: {n:,} lines, so {n - 1:,} rows")

Things to notice: the timestamps look like 2024-06-01 00:05:00: they have no time zone, and the values are in quotes. The pipeline assumes UTC. (That is an assumption: ask yourself what would be wrong if the times were really Irish local time.)
Tasks: compare the files with what the web page says

if RUN_REAL:
    june = pd.read_csv("data/raw/dublin-bikes_station_status_062024.csv", nrows=0)
    feb  = pd.read_csv("data/raw/dublin-bikes_station_status_022026.csv", nrows=0)
    same_columns = ___            # True or False: do the two files have exactly the same columns?
    june_rows    = ___            # how many data rows does the June 2024 file have? (lines minus the header)
    print(len(june.columns), "columns in June,", len(feb.columns), "in February;", f"June rows: {june_rows:,}")

if RUN_REAL:
    check("both files have the same columns", same_columns is True)
    check("the June 2024 file has 597,669 rows", june_rows == 597669)

Your answer (2 or 3 sentences). The dataset’s web page says the June 2024 file has 315,747 rows and that the February 2026 file has only 9 columns. What do you actually find? What does that teach you about trusting a description of a dataset?

Write your answer here (double-click to edit, then press Shift + Enter).
Step 3: run the pipeline on the real files

Same pipeline, same command, bigger files. On a recent laptop this takes about 20 to 60 seconds; an older machine takes longer. Wait for the output.

if RUN_REAL:
    shutil.rmtree("out/real", ignore_errors=True)
    code, real_counters, seconds = run_pipeline("data/raw/*.csv", "data/ref/dublin.csv", "out/real/hourly")
    print(f"\nExit code {code}; took {seconds:.0f} seconds")

You should see counters like these (the exact numbers are for the files as published when the lecture was written; if yours are very different, the council may have updated a file: tell your tutor):

  files_read      2
  good_rows       1052292
  rows_read       1052292
  unknown_station 6828

There is no bad_rows line: a counter that never fires is not printed, so a missing line means 0. The real data is very clean.

if RUN_REAL:
    check("two files were read", real_counters.get("files_read") == 2)
    check("no row was lost: rows_read equals good_rows plus bad_rows", real_counters["rows_read"] == real_counters["good_rows"] + real_counters.get("bad_rows", 0))
    check("nothing was quarantined on the real data", real_counters.get("bad_rows", 0) == 0)
    print("unknown_station:", real_counters.get("unknown_station", 0), "(we will investigate this in Exercise 2)")

Step 4: check the pipeline against a second, independent method

A pipeline you cannot check is a pipeline you cannot trust. The script reconcile_local.py recomputes the same hourly numbers with plain pandas, from the raw files, and compares them with the pipeline’s output. If two completely different methods agree, both are very probably right.

if RUN_REAL:
    code, out = run([PY, "reconcile_local.py"])
    check("the two methods agree on every row", "All three equal" in out)

Step 5: count the rows per day

The script rows_per_day.py counts how many rows arrive each day. We use the same idea here with a chart. Complete the one blank: a day is “thin” when it has fewer than 75% of the rows of a reference day (we use a busy day: the 90th percentile).

if RUN_REAL:
    import matplotlib.pyplot as plt
    june_times = pd.read_csv("data/raw/dublin-bikes_station_status_062024.csv", usecols=["last_reported"], dtype=str)["last_reported"]
    try:
        when = pd.to_datetime(june_times, errors="coerce", utc=True, format="ISO8601")
    except (TypeError, ValueError):               # older versions of pandas
        when = pd.to_datetime(june_times, errors="coerce", utc=True)
    per_day = when.dropna().dt.date.value_counts().sort_index()
    reference_day = per_day.quantile(0.90)
    threshold = ___                               # 75% of the reference day
    thin_days = per_day[per_day < threshold]
    colours = ["#E64141" if day in thin_days.index else "#3431B6" for day in per_day.index]
    ax = per_day.plot.bar(color=colours, figsize=(11, 3.5))
    ax.set_title("Rows per day, June 2024 (thin days in red)"); ax.set_ylabel("rows"); plt.tight_layout(); plt.show()
    print("Reference day:", round(reference_day), "rows; thin days:", [str(d) for d in thin_days.index])

Hint

if RUN_REAL:
    check("exactly two thin days were found", len(thin_days) == 2)
    check("they are 5 and 6 June 2024", {str(d) for d in thin_days.index} == {"2024-06-05", "2024-06-06"})

Your answer (2 or 3 sentences). Describe what the chart shows: what is a normal day, which days stand out, and do weekends look different? What would you ask Dublin City Council about the thin days? (The data can tell you when, never why.)

Write your answer here (double-click to edit, then press Shift + Enter).
Part 7. Exercise 2: investigate the real results (15 min)

In Part 6 one counter was not zero: unknown_station. It counts hourly results whose station is not in the station list. In this exercise you find out which stations are affected, using pandas on the pipeline’s real output.

Reminder: the output of the pipeline is in out/real/hourly-*.jsonl: one line per station per hour. The read_jsonl function from Part 4 loads it into a table.
Task A: load the real results

if RUN_REAL:
    hourly_real = ___             # load every file that matches "out/real/hourly-0*.jsonl" (use read_jsonl)
    print(hourly_real.shape)
    display(hourly_real.head(3))

Hint

if RUN_REAL:
    check("143,431 hourly rows (the same as the reconciliation said)", len(hourly_real) == 143431)

Task B: which stations have no name?

if RUN_REAL:
    unnamed = hourly_real[___]                      # keep only the rows where station_name is missing
    unknown_ids = set(___)                          # the set of different station ids in those rows
    print(len(unnamed), "hourly rows with no station name")
    print("station ids:", sorted(unknown_ids))

Hint

if RUN_REAL:
    check("6,828 rows have no name, matching the unknown_station counter", len(unnamed) == real_counters.get("unknown_station"))
    check("the six station ids are 1, 14, 20, 35, 60 and 70", unknown_ids == {1, 14, 20, 35, 60, 70})

Task C: when do they appear?

Count the hourly rows per month for the whole output, and then for the unnamed stations only. (hour_start is text like 2024-06-01 00:00:00 UTC: its first seven characters are the month.)

if RUN_REAL:
    month = hourly_real["hour_start"].astype(str).str[:7]
    rows_per_month = month.value_counts().sort_index()                 # hourly rows in each month (given)
    unnamed_per_month = ___                                            # the same, but only for the unnamed stations
    print(rows_per_month)
    print()
    print(unnamed_per_month)

Hint

if RUN_REAL:
    check("the unnamed rows appear in both months", set(unnamed_per_month.index) == {"2024-06", "2026-02"})

Your answer (3 or 4 sentences). (1) Five of the six stations appear in both months and station 70 only in February 2026. What are two possible reasons why the station list does not contain them? (2) The pipeline kept these rows and counted them instead of dropping them. Why is that the better choice? (3) What would you ask the data owner?

Write your answer here (double-click to edit, then press Shift + Enter).

Your answer (2 or 3 sentences). The timestamps in the files have no time zone, and our pipeline assumed they are UTC and labelled the hours UTC. If the times were really Irish local time (one hour ahead of UTC in summer), what would be wrong with the results? How could you find out?

Write your answer here (double-click to edit, then press Shift + Enter).
Part 8. Wrap-up (5 min)
What you did
Step 	What it showed
Read the code 	A pipeline is a chain of small, named steps; the data contract runs on every row
Ran the tests 	A pipeline is code and can be tested before it touches real data
Ran the sample 	The counters are the pipeline’s test report; the quarantine keeps bad rows with a reason
Exercise 1 	You can predict what a pipeline will do from its rules
Ran the real data 	The same code handles a million rows in under a minute, and nothing was lost
Reconciliation 	Two different methods agreeing is strong evidence the pipeline is right
Exercise 2 	A counter that fires on real data points you to a real data problem
Five ideas to remember

    Look at the data before you run anything on it.
    Count every row: rows_read = good + bad, always.
    Quarantine, do not crash and do not hide.
    A description of a dataset is not the data (the web page was wrong twice).
    Check a result with a second method.

Clean up (optional)

The cell below deletes the files this notebook created (the sample, your exercise, the outputs and the downloaded data) so they do not take up disk space. Set REALLY_DELETE = True to do it.

REALLY_DELETE = False       # change to True to delete everything this notebook created

if REALLY_DELETE:
    for folder in ["sample", "exercise1", "out", "data"]:
        shutil.rmtree(folder, ignore_errors=True)
        print("deleted", folder)
else:
    print("Nothing deleted. (Set REALLY_DELETE = True to clean up.)")

What comes next

In the lab you will run the same pipeline on Google Cloud Dataflow, and extend it with three small changes of your own. Everything you did here (reading the counters, the quarantine, the reconciliation) works the same way there.
Glossary
Word 	Meaning
Pipeline 	The steps that take raw data to a trustworthy result, automatically
Apache Beam 	The programming model (and Python library) used to write the pipeline
DirectRunner 	Beam’s runner that works on your own computer
PCollection 	A dataset inside the pipeline
DoFn 	A class that defines one step, with a process method
Counter 	A number the pipeline keeps to report what happened
Quarantine 	Where rows that break the rules are kept, with the reason
Window 	A slice of time (here, one hour) used to group readings
Data contract 	The rules the data must follow, written as code
Reconciliation 	Checking that two different methods give the same answer
Schema drift 	The columns of a file changing over time