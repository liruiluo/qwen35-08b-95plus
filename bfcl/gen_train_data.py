#!/usr/bin/env python3
"""Generate synthetic BFCL-simple_python-style SFT training data.

The baseline audit showed the model's failures cluster into convention families
(percent-vs-decimal rates, canonical tokens, date formats, string-typed
numerics, dict inner keys, canonical list elements). This generator builds
training items that teach those conventions through realistic schemas whose
descriptions state the expected format, mirroring how the official BFCL golds
behave. It does NOT copy any real BFCL v4 item (a dedup guard drops any
collision).

Each family mixes the target convention with look-alike distractors so the
model must read the schema instead of applying one blanket rule.

Output:
  train.jsonl  (messages format, assistant = Qwen tool markup)
  val.jsonl    (held-out 4%)
  manifest.json (per-family counts)

Usage: python gen_train_data.py --out-dir data [--seed 7]
"""
import argparse, json, os, random

# ---------------------------------------------------------------- pools
FIRST = ["Alex", "Sam", "Robert", "Steve", "Emma", "Liam", "Olivia", "Noah",
         "Ava", "Ethan", "Mia", "Lucas", "Isabella", "Mason", "Sophia", "Logan"]
CITIES = [("New York City, NY", ["Manhattan, New York", "NYC", "New York"]),
          ("San Francisco, CA", ["SF", "San Francisco"]),
          ("Chicago, IL", ["Chicago"]),
          ("Boston, MA", ["Boston"]),
          ("Seattle, WA", ["Seattle"]),
          ("Austin, TX", ["Austin"]),
          ("Denver, CO", ["Denver"]),
          ("Miami, FL", ["Miami"])]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
FREQ_WORD = {"monthly": "month", "annually": "year", "quarterly": "quarter",
             "semi-annually": "half year", "daily": "day"}
COMPOUND_N = {"monthly": 12, "annually": 1, "quarterly": 4,
              "semi-annually": 2, "daily": 365}

SPECIES = {"human": "Homo sapiens", "dog": "Canis lupus familiaris",
           "cat": "Felis catus", "mouse": "Mus musculus",
           "fruit fly": "Drosophila melanogaster", "yeast": "Saccharomyces cerevisiae",
           "chicken": "Gallus gallus", "rice": "Oryza sativa"}
ROOM_TYPES = ["single", "double", "queen", "king", "suite", "deluxe", "twin"]
ENERGY = ["renewable", "non-renewable", "solar", "wind", "hydroelectric",
          "nuclear", "natural gas", "coal", "biomass", "geothermal"]
PLANT_TYPES = ["Annual", "Perennial", "Biennial", "Evergreen", "Deciduous"]
LIB_SECTIONS = ["Fiction", "Non-Fiction", "Science", "Children's", "Romance",
                "Mystery", "History", "Biography", "Poetry", "Reference"]
MUSEUM_INFO = ["opening_hours", "ticket_price", "location", "history",
               "exhibits", "tours"]
BOARDGAME_PARAMS = ["rating", "player count", "playtime", "age range",
                    "publisher", "designer"]
FINISHES = ["Rosewood", "Oak", "Walnut", "Mahogany", "Maple", "Cherry", "Pine"]
GENDERS = ["men", "women", "unisex", "boys", "girls"]
DEPARTMENTS = ["Pediatrics", "Cardiology", "Oncology", "Neurology", "Dermatology",
               "Orthopedics", "Radiology", "Emergency", "Psychiatry", "Urology"]
MEALS = ["breakfast", "lunch", "dinner", "brunch", "supper", "snack"]
WEATHER_DETAILS = ["high temperature", "low temperature", "humidity",
                   "precipitation", "wind speed", "UV index"]
GROCERY_CATEGORIES = ["Fruits", "Vegetables", "Dairy", "Bakery", "Meat",
                      "Seafood", "Snacks", "Beverages", "Frozen", "Organic"]
NUTRITION = ["Protein", "Calories", "Carbohydrates", "Fat", "Fiber",
             "Sugar", "Sodium", "Vitamins"]
CANONICAL_ITEMS = ["bottled water", "apples", "rice", "milk", "eggs", "bread",
                   "chicken breast", "pasta", "cereal", "orange juice",
                   "peanut butter", "yogurt", "bananas", "coffee", "flour"]
ITEM_PARAPHRASE = {
    "bottled water": ["a 12 pack of bottled water", "some bottled water",
                      "a pack of bottled water", "bottled waters"],
    "apples": ["a bag of apples", "some apples", "red apples", "fresh apples"],
    "eggs": ["a dozen eggs", "some eggs", "eggs"],
    "milk": ["a gallon of milk", "some milk", "two percent milk"],
    "rice": ["a bag of rice", "white rice", "some rice"],
    "bread": ["a loaf of bread", "sliced bread", "some bread"],
    "coffee": ["a bag of ground coffee", "ground coffee", "some coffee"],
    "bananas": ["a bunch of bananas", "some bananas"],
    "yogurt": ["a cup of yogurt", "some yogurt", "greek yogurt"],
    "cereal": ["a box of cereal", "some cereal"],
    "orange juice": ["a carton of orange juice", "some orange juice"],
    "chicken breast": ["boneless chicken breast", "a pack of chicken breast"],
    "pasta": ["a box of pasta", "some pasta", "spaghetti"],
    "peanut butter": ["a jar of peanut butter", "some peanut butter"],
    "flour": ["a bag of flour", "some flour", "all purpose flour"],
    "cereal ": ["a box of cereal"]}

DB_TABLES = {
    "Students": (["name", "school", "subject", "grade", "age"],
                 {"school": ["Bluebird High School", "Lincoln High School",
                             "Washington Elementary", "Riverside Middle School"],
                  "subject": ["Science", "Math", "History", "English",
                              "Biology", "Chemistry", "Physics"]}),
    "Employees": (["name", "department", "salary", "years_of_service", "office"],
                  {"department": ["Engineering", "Marketing", "Sales",
                                  "Human Resources", "Finance"],
                   "office": ["New York", "San Francisco", "Austin", "Seattle"],
                   "title": ["Engineer", "Manager", "Analyst", "Director"]}),
    "Patients": (["name", "doctor", "diagnosis", "age"],
                 {"doctor": ["Dr. Smith", "Dr. Johnson", "Dr. Williams"],
                  "diagnosis": ["flu", "hypertension", "diabetes"]}),
}

# ---------------------------------------------------------------- helpers

def pnum(x):
    """Pretty number: int-valued floats -> int."""
    if isinstance(x, float) and x.is_integer():
        return int(x)
    return x


def markup(name, params):
    lines = [f"<function={name}>"]
    for k, v in params.items():
        lines.append(f"<parameter={k}>")
        lines.append(json.dumps(v, ensure_ascii=False))
        lines.append("</parameter>")
    lines.append("</function>")
    return "\n".join(lines)


def item(question, fname, schema_params, required, gold_params, item_id):
    """Build one training item. gold_params: {param: allowed-list}."""
    schema = {"name": fname,
              "description": schema_params.pop("_desc"),
              "parameters": {"type": "object",
                             "properties": schema_params,
                             "required": required}}
    return {"id": item_id,
            "question": question,
            "function": [schema],
            "ground_truth": [{fname: gold_params}]}


def to_oai_tool(fd):
    """Same conversion the eval harness applies before sending to the server."""
    props = dict(fd["parameters"]["properties"])
    req = fd["parameters"].get("required", [])
    schema = {"type": "object", "properties": {}, "required": req}
    for k, v in props.items():
        schema["properties"][k] = {"type": v.get("type", "string"),
                                  "description": v.get("description", "")}
    return {"type": "function",
            "function": {"name": fd["name"].replace(".", "_"),
                         "description": fd.get("description", ""),
                         "parameters": schema}}


def to_sft(it, system):
    q = it["question"]
    fname = it["function"][0]["name"]
    target_name = fname.replace(".", "_")
    gt = it["ground_truth"][0][fname]
    # target call: first allowed value for each param (in gold order)
    params = {}
    for k, allowed in gt.items():
        vals = [v for v in allowed if v != ""]
        if vals:
            params[k] = vals[0]
    return {"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": q},
        {"role": "assistant", "content": markup(target_name, params)}],
        "tools": [to_oai_tool(it["function"][0])]}


# ---------------------------------------------------------------- families

def fam_rate(rng, i):
    """Percent-in-question -> decimal rate; freq word vs periods-per-year."""
    name = rng.choice(["compound_interest", "calculate_loan_interest",
                       "finance_calculator_npv", "finance_predict_future_value",
                       "investment_predictProfit", "calculate_future_value",
                       "calculate_discounted_cash_flow", "savings_projection"])
    rate_key = rng.choice(["annual_rate", "interest_rate", "discount_rate",
                           "annual_interest_rate", "annual_return",
                           "rate_of_return", "annual_yield", "annual_growth_rate"])
    principal = rng.choice([500, 1000, 2500, 5000, 8000, 10000, 12000, 15000,
                            20000, 25000, 50000])
    pct = rng.choice([2, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 9, 10, 12])
    rate = pct / 100
    years = rng.randint(1, 10)
    freq = rng.choice(list(FREQ_WORD))
    principal_d = rng.choice([100, 200, 250, 500, 1000])
    period = rng.randint(3, 10)

    if name in ("finance_calculator_npv", "calculate_discounted_cash_flow"):
        q = (f"Calculate the net present value of an investment with an initial cost "
             f"of ${-principal} and cash flows of ${principal_d} per period for {period} "
             f"periods, using a discount rate of {pct}% per period.")
        gold = {rate_key: [rate], "period": [period],
                "cash_flows": [[pnum(-principal)] + [principal_d] * period]}
        props = {rate_key: {"type": "float", "description": "Discount rate per period as a decimal fraction (e.g. 0.05 for 5%)"},
                 "period": {"type": "integer"},
                 "cash_flows": {"type": "array", "items": {"type": "float"},
                                "description": "Cash flow per period, starting with the initial investment (negative)"}}
        req = [rate_key, "period", "cash_flows"]
    else:
        if rng.random() < 0.5:
            q = (f"Calculate the {name.replace('_', ' ')} on a principal of "
                 f"${principal} at an annual interest rate of {pct}% compounded "
                 f"{freq} for {years} years.")
            gold = {"principal": [principal], rate_key: [rate],
                    "compounding_freq": [freq], "time_years": [years]}
            props = {"principal": {"type": "float"},
                     rate_key: {"type": "float", "description": "Annual interest rate as a decimal fraction (e.g. 0.05 for 5%)"},
                     "compounding_freq": {"type": "string", "description": "Compounding frequency as a word: monthly, quarterly, semi-annually, annually, or daily"},
                     "time_years": {"type": "integer"}}
            req = ["principal", rate_key, "compounding_freq", "time_years"]
        else:
            n_per_year = COMPOUND_N[freq]
            q = (f"Predict the future value of an investment of ${principal} with an "
                 f"annual return of {pct}%, compounded {freq}, held for {years} years.")
            gold = {"present_value": [principal], rate_key: [rate],
                    "compounding_periods_per_year": [n_per_year],
                    "time_years": [years]}
            props = {"present_value": {"type": "float"},
                     rate_key: {"type": "float", "description": "Annual rate as a decimal fraction (e.g. 0.05 for 5%)"},
                     "compounding_periods_per_year": {"type": "integer", "description": "Number of compounding periods per year (12 for monthly, 4 for quarterly, 1 for annually)"},
                     "time_years": {"type": "integer"}}
            req = ["present_value", rate_key, "compounding_periods_per_year", "time_years"]

    # distractor family: same shape but the schema says PERCENT -> keep 5 not 0.05
    if rng.random() < 0.25:
        pct_key = "rate_percent"
        q = (f"What is the simple interest earned on a deposit of ${principal} "
             f"at an annual rate of {pct} percent for {years} years? "
             f"Use the simple_interest_calculator.")
        gold = {"principal": [principal], pct_key: [pct], "time_years": [years]}
        props = {"principal": {"type": "float"},
                 pct_key: {"type": "float", "description": "Annual rate in percent (e.g. 5.0 means 5%, NOT 0.05)"},
                 "time_years": {"type": "integer"}}
        req = ["principal", pct_key, "time_years"]
        name = "simple_interest_calculator"
    return item(q, name, {"_desc": f"Calculate {name.replace('_',' ')} for a financial quantity.", **props},
                req, gold, f"f_rate_{i}")


def fam_canonical(rng, i):
    kind = rng.choice(["species", "room", "energy", "museum", "boardgame",
                       "finish", "gender", "department", "library"])
    if kind == "species":
        common, latin = rng.choice(list(SPECIES.items()))
        gene = rng.choice(["BRCA1", "TP53", "EGFR", "KRAS", "APOE", "MYC", "TNF"])
        q = f"Get the protein sequence for the gene {gene} in {common}s."
        gold = {"gene": [gene], "species": [latin]}
        props = {"gene": {"type": "string"},
                 "species": {"type": "string", "description": "Species in Latin binomial nomenclature, e.g. Homo sapiens for human"}}
        req = ["gene", "species"]; name = "get_protein_sequence"
    elif kind == "room":
        rt = rng.choice(ROOM_TYPES)
        city, _ = rng.choice(CITIES)
        nights = rng.randint(1, 7)
        q = f"Book a {rt} room in {city} for {nights} nights."
        gold = {"room_type": [rt], "location": [city], "nights": [nights]}
        props = {"room_type": {"type": "string", "description": "one of: single, double, queen, king, suite, deluxe, twin"},
                 "location": {"type": "string"}, "nights": {"type": "integer"}}
        req = ["room_type", "location", "nights"]; name = "hotel_booking"
    elif kind == "energy":
        e = rng.choice(ENERGY)
        country = rng.choice(["Germany", "France", "United States", "China",
                              "Brazil", "India", "Japan", "Canada"])
        q = f"Get statistics on {e} energy production in {country} for the last five years."
        gold = {"energy_type": [e], "country": [country], "time_frame": ["five_years"]}
        props = {"energy_type": {"type": "string", "description": "one of: renewable, non-renewable, solar, wind, hydroelectric, nuclear, natural gas, coal, biomass, geothermal"},
                 "country": {"type": "string"},
                 "time_frame": {"type": "string", "description": "one of: one_year, five_years, ten_years, all_time"}}
        req = ["energy_type", "country", "time_frame"]; name = "energy_production_stats"
    elif kind == "museum":
        mi = rng.choice(MUSEUM_INFO)
        museum = rng.choice(["Metropolitan Museum of Art", "Louvre Museum",
                             "British Museum", "Prado Museum",
                             "National Gallery of Art", "Van Gogh Museum"])
        q = f"Find the {mi.replace('_',' ')} for the {museum}."
        gold = {"museum_name": [museum], "info_type": [mi]}
        props = {"museum_name": {"type": "string"},
                 "info_type": {"type": "string", "description": "one of: opening_hours, ticket_price, location, history, exhibits, tours"}}
        req = ["museum_name", "info_type"]; name = "museum_info"
    elif kind == "boardgame":
        bg = rng.choice(["Ticket to Ride", "Catan", "Pandemic", "Carcassonne",
                         "Dominion", "Wingspan", "Azul", "7 Wonders"])
        k = rng.randint(1, len(BOARDGAME_PARAMS))
        wanted = rng.sample(BOARDGAME_PARAMS, k)
        q = f"Look up {', '.join(wanted)} for the board game {bg}."
        gold = {"name": [bg], "parameters": [wanted]}
        props = {"name": {"type": "string"},
                 "parameters": {"type": "array", "items": {"type": "string"},
                                "description": "which info to fetch; each element must be one of: rating, player count, playtime, age range, publisher, designer"}}
        req = ["name", "parameters"]; name = "boardgame.get_info"
    elif kind == "finish":
        fin = rng.choice(FINISHES)
        furniture = rng.choice(["dining table", "bookshelf", "desk", "cabinet",
                                "coffee table", "bed frame"])
        q = f"I'm looking for a {furniture} with a {fin} finish."
        gold = {"furniture_type": [furniture], "finish": [fin]}
        props = {"furniture_type": {"type": "string"},
                 "finish": {"type": "string", "description": "one of: Rosewood, Oak, Walnut, Mahogany, Maple, Cherry, Pine"}}
        req = ["furniture_type", "finish"]; name = "furniture_search"
    elif kind == "gender":
        g = rng.choice(GENDERS)
        color = rng.choice(["blue", "black", "white", "red", "gray", "green"])
        size = rng.choice([7, 8, 9, 10, 11])
        q = f"Show me {color} {g}'s running shoes in size {size}."
        gold = {"gender": [g], "color": [color], "size": [size]}
        props = {"gender": {"type": "string", "description": "one of: men, women, unisex, boys, girls"},
                 "color": {"type": "string"}, "size": {"type": "integer"}}
        req = ["gender", "color", "size"]; name = "shoes_search"
    elif kind == "department":
        dep = rng.choice(DEPARTMENTS)
        doc = rng.choice(["Dr. Smith", "Dr. Johnson", "Dr. Williams", "Dr. Brown"])
        q = f"Find the schedule for the {dep} department with {doc}."
        gold = {"department": [dep], "doctor": [doc]}
        props = {"department": {"type": "string", "description": "one of: Pediatrics, Cardiology, Oncology, Neurology, Dermatology, Orthopedics, Radiology, Emergency, Psychiatry, Urology"},
                 "doctor": {"type": "string"}}
        req = ["department", "doctor"]; name = "hospital_department_schedule"
    else:
        sec = rng.choice(LIB_SECTIONS)
        lib = rng.choice(["Central Library", "Riverside Branch Library",
                          "Eastside Public Library", "University Library"])
        q = f"What new arrivals are in the {sec} section of the {lib}?"
        gold = {"library_name": [lib], "section": [sec]}
        props = {"library_name": {"type": "string"},
                 "section": {"type": "string", "description": "one of: Fiction, Non-Fiction, Science, Children's, Romance, Mystery, History, Biography, Poetry, Reference"}}
        req = ["library_name", "section"]; name = "library_new_arrivals"
    return item(q, name, {"_desc": f"Query {name.replace('_',' ')} information.", **props}, req, gold, f"f_tok_{i}")


def fam_date(rng, i):
    m = rng.randint(1, 12)
    mm = MONTHS[m - 1]
    d = rng.randint(1, 28)
    y = rng.choice([2021, 2022, 2023, 2024])
    md = f"{m:02d}-{d:02d}-{y}"
    iso = f"{y}-{m:02d}-{d:02d}"
    natural = f"{mm} {d}, {y}"
    phr = rng.choice([natural, f"{mm} {d}th, {y}", f"the {d}th of {mm} {y}",
                      f"{d} {mm} {y}", iso])
    kind = rng.random()
    if kind < 0.5:
        q = f"Find all events on {phr}."
        gold = {"date": [md], "event_type": ["all"]}
        props = {"date": {"type": "string", "description": "date in MM-DD-YYYY format"},
                 "event_type": {"type": "string", "description": "one of: all, concert, sports, theater"}}
        req = ["date", "event_type"]; name = "events_search"
    elif kind < 0.8:
        q = f"Get the sales report for {phr}."
        gold = {"report_date": [iso], "report_type": ["sales"]}
        props = {"report_date": {"type": "string", "description": "date in YYYY-MM-DD format"},
                 "report_type": {"type": "string", "description": "one of: sales, revenue, inventory"}}
        req = ["report_date", "report_type"]; name = "get_report"
    else:
        hour = rng.randint(1, 12)
        h24 = hour if hour < 10 else hour  # keep simple
        ampm = rng.choice(["AM", "PM"])
        h_val = hour if ampm == "AM" else hour + 12
        q = f"Book the conference room on {phr} at {hour} {ampm}."
        gold = {"date": [md], "time": [f"{h_val:02d}:00"]}
        props = {"date": {"type": "string", "description": "date in MM-DD-YYYY format"},
                 "time": {"type": "string", "description": "time in 24-hour HH:MM format"}}
        req = ["date", "time"]; name = "room_reservation"
    return item(q, name, {"_desc": f"Look up {name.replace('_',' ')}.", **props}, req, gold, f"f_date_{i}")


def fam_strnum(rng, i):
    year = rng.choice([2019, 2020, 2021, 2022, 2023])
    num = rng.randint(100000, 999999)
    kind = rng.choice(["year", "season", "docket", "patient", "parcel", "zip"])
    if kind == "year":
        q = f"Get the annual summary for {year}."
        gold = {"year": [str(year)]}
        props = {"year": {"type": "string", "description": "the year as a string, e.g. \"2022\""}}
        req = ["year"]; name = "annual_summary"
    elif kind == "season":
        season_year = rng.choice(["2021", "2022", "2023"])
        league = rng.choice(["NBA", "NFL", "MLB", "NHL", "Premier League"])
        q = f"Show me the championship results for the {league} {season_year} season."
        gold = {"league": [league], "season": [season_year]}
        props = {"league": {"type": "string"},
                 "season": {"type": "string", "description": "season year as a string, e.g. \"2021\""}}
        req = ["league", "season"]; name = "championship_results"
    elif kind == "docket":
        q = f"Retrieve the court record with docket number {num}."
        gold = {"docket_number": [str(num)]}
        props = {"docket_number": {"type": "string", "description": "docket number as a string of digits"}}
        req = ["docket_number"]; name = "court_record_lookup"
    elif kind == "patient":
        pid = rng.randint(100000, 999999)
        q = f"Pull up the medical record for patient ID {pid}."
        gold = {"patient_id": [str(pid)]}
        props = {"patient_id": {"type": "string", "description": "patient identifier as a string of digits"}}
        req = ["patient_id"]; name = "medical_record_lookup"
    elif kind == "parcel":
        pn = rng.randint(100000000, 999999999)
        q = f"Look up the property with parcel number {pn}."
        gold = {"parcel_number": [str(pn)]}
        props = {"parcel_number": {"type": "string", "description": "parcel number as a string of digits"}}
        req = ["parcel_number"]; name = "property_lookup"
    else:
        zipc = rng.choice(["10001", "94105", "60601", "02110", "98101",
                           "78701", "80202", "33101"])
        q = f"Find all service centers in zip code {zipc}."
        gold = {"zip_code": [zipc]}
        props = {"zip_code": {"type": "string", "description": "5-digit zip code as a string"}}
        req = ["zip_code"]; name = "service_centers_by_zip"
    return item(q, name, {"_desc": f"Look up {name.replace('_',' ')}.", **props}, req, gold, f"f_str_{i}")


def fam_dict(rng, i):
    table = rng.choice(list(DB_TABLES))
    cols, pools = DB_TABLES[table]
    if rng.random() < 0.5:
        # flat dict conditions
        k1, k2 = rng.sample(list(pools), 2)
        v1, v2 = rng.choice(pools[k1]), rng.choice(pools[k2])
        q = (f"Fetch records from the {table} table where {k1} is {v1} "
             f"and {k2} is {v2}.")
        gold = {"table_name": [table],
                "conditions": [{k1: [v1], k2: [v2]}]}
        props = {"table_name": {"type": "string"},
                 "conditions": {"type": "dict",
                                "description": f"exact keys drawn from: {', '.join(pools)}"}}
        req = ["table_name", "conditions"]; name = "db_fetch_records"
    else:
        # list-of-dicts with column/operator/value
        c1, c2 = rng.sample(cols, 2)
        op1 = rng.choice([">", "<", ">=", "<=", "="])
        val1 = rng.randint(18, 65)
        eqval = rng.choice(pools[list(pools)[0]])
        eqcol = list(pools)[0]
        q = (f"Query the {table} table for records where {c1} {op1} {val1} "
             f"and {eqcol} equals {eqval}.")
        gold = {"table": [table],
                "conditions": [[{"column": [c1], "operator": [op1], "value": [val1]},
                                {"column": [eqcol], "operator": ["="], "value": [eqval]}]]}
        props = {"table": {"type": "string"},
                 "conditions": {"type": "array", "items": {"type": "dict"},
                                "description": "each element must have exactly the keys: column, operator, value"}}
        req = ["table", "conditions"]; name = "database_query"
    return item(q, name, {"_desc": "Query a database.", **props}, req, gold, f"f_dict_{i}")


def fam_list(rng, i):
    kind = rng.choice(["grocery", "nutrition", "weather", "shopping"])
    if kind == "grocery":
        k = rng.randint(2, 4)
        cats = rng.sample(GROCERY_CATEGORIES, k)
        store = rng.choice(["Whole Foods", "Trader Joe's", "Safeway", "Kroger"])
        q = f"What {', '.join(cats)} products does {store} carry?"
        gold = {"store_name": [store], "categories": [cats]}
        props = {"store_name": {"type": "string"},
                 "categories": {"type": "array", "items": {"type": "string"},
                                "description": f"each element must be one of: {', '.join(GROCERY_CATEGORIES)}"}}
        req = ["store_name", "categories"]; name = "grocery_inventory"
    elif kind == "nutrition":
        k = rng.randint(2, 4)
        info = rng.sample(NUTRITION, k)
        food = rng.choice(["apple", "banana", "chicken breast", "rice",
                           "avocado", "salmon", "egg", "oatmeal"])
        q = f"Get the {', '.join(info).lower()} for {food}."
        gold = {"food": [food], "information": [info]}
        props = {"food": {"type": "string"},
                 "information": {"type": "array", "items": {"type": "string"},
                                 "description": f"each element must be one of: {', '.join(NUTRITION)}"}}
        req = ["food", "information"]; name = "nutrition_info"
    elif kind == "weather":
        k = rng.randint(2, 4)
        det = rng.sample(WEATHER_DETAILS, k)
        city, _ = rng.choice(CITIES)
        q = f"Give me the {', '.join(det)} for {city} today."
        gold = {"location": [city], "details": [det]}
        props = {"location": {"type": "string"},
                 "details": {"type": "array", "items": {"type": "string"},
                             "description": f"each element must be one of: {', '.join(WEATHER_DETAILS)}"}}
        req = ["location", "details"]; name = "weather_details"
    else:
        k = rng.randint(2, 4)
        wanted = rng.sample(CANONICAL_ITEMS, k)
        phrases = [rng.choice(ITEM_PARAPHRASE.get(w, [w])) for w in wanted]
        store = rng.choice(["Costco", "Walmart", "Target"])
        q = f"Check if {store} has {' and '.join(phrases)} in stock."
        gold = {"store": [store], "items": [wanted]}
        props = {"store": {"type": "string"},
                 "items": {"type": "array", "items": {"type": "string"},
                           "description": f"canonical item names; each element must be one of: {', '.join(CANONICAL_ITEMS)}"}}
        req = ["store", "items"]; name = "check_stock"
    return item(q, name, {"_desc": f"Get {name.replace('_',' ')}.", **props}, req, gold, f"f_list_{i}")


def fam_plain(rng, i):
    kind = rng.choice(["triangle", "factorial", "hypot", "convert", "lookup"])
    if kind == "triangle":
        b, h = rng.randint(2, 50), rng.randint(2, 50)
        q = f"Find the area of a triangle with a base of {b} units and height of {h} units."
        gold = {"base": [b], "height": [h]}
        props = {"base": {"type": "integer"}, "height": {"type": "integer"}}; req = ["base", "height"]
        name = "calculate_triangle_area"
    elif kind == "factorial":
        n = rng.randint(2, 15)
        q = f"What is the factorial of {n}?"
        gold = {"number": [n]}
        props = {"number": {"type": "integer"}}; req = ["number"]
        name = "math.factorial"
    elif kind == "hypot":
        x, y = rng.randint(1, 30), rng.randint(1, 30)
        z = rng.choice([0, None])
        q = f"Compute the Euclidean norm of a {x} by {y} by {z if z else 0} box."
        gold = {"x": [x], "y": [y], "z": [z if z is not None else 0]}
        props = {"x": {"type": "integer"}, "y": {"type": "integer"},
                 "z": {"type": "integer", "description": "third dimension, 0 if not applicable"}}
        req = ["x", "y", "z"]; name = "math.hypot"
    elif kind == "convert":
        v = rng.choice([1, 5, 10, 20, 50, 100, 250])
        pair = rng.choice([("USD", "EUR"), ("USD", "JPY"), ("EUR", "GBP"),
                           ("km", "miles"), ("kg", "lbs"), ("C", "F")])
        q = f"Convert {v} {pair[0]} to {pair[1]}."
        gold = {"value": [v], "from_unit": [pair[0]], "to_unit": [pair[1]]}
        props = {"value": {"type": "float"}, "from_unit": {"type": "string"},
                 "to_unit": {"type": "string"}}
        req = ["value", "from_unit", "to_unit"]; name = "unit_converter"
    else:
        first = rng.choice(FIRST)
        q = f"Look up the contact information for {first} in the employee directory."
        gold = {"name": [first]}
        props = {"name": {"type": "string", "description": "full first name"}}; req = ["name"]
        name = "directory_lookup"
    return item(q, name, {"_desc": f"Compute or look up {name.replace('.','_').replace('_',' ')}.", **props}, req, gold, f"f_plain_{i}")


def fam_optional(rng, i):
    """Precision/optionality: flags default false, round_to 2, optional omitted."""
    kind = rng.choice(["flag_off", "flag_on", "round", "omit_optional"])
    n = rng.randint(2, 500)
    if kind == "flag_off":
        q = f"List the movies currently playing in theaters."
        gold = {"category": ["in_theaters"], "include_upcoming": [False]}
        props = {"category": {"type": "string", "description": "one of: in_theaters, upcoming, top_rated, new_releases"},
                 "include_upcoming": {"type": "boolean", "description": "include upcoming releases; default false"}}
        req = ["category"]; name = "movies_list"
    elif kind == "flag_on":
        q = f"List the movies currently playing in theaters, including upcoming releases."
        gold = {"category": ["in_theaters"], "include_upcoming": [True]}
        props = {"category": {"type": "string", "description": "one of: in_theaters, upcoming, top_rated, new_releases"},
                 "include_upcoming": {"type": "boolean"}}
        req = ["category", "include_upcoming"]; name = "movies_list"
    elif kind == "round":
        a, b = rng.randint(2, 40), rng.randint(2, 40)
        q = f"What is the probability of drawing a specific card from a standard deck of 52 cards, drawing {a} times?"
        gold = {"total_outcomes": [52], "favorable_outcomes": [4],
                "round_to": ["", 2]}
        props = {"total_outcomes": {"type": "integer"}, "favorable_outcomes": {"type": "integer"},
                 "round_to": {"type": "integer", "description": "decimal places to round to; default 2"}}
        req = ["total_outcomes", "favorable_outcomes"]; name = "calculate_probability"
    else:
        first = rng.choice(FIRST)
        q = f"Get the profile for user {first}."
        gold = {"username": [first], "include_email": ["", False]}
        props = {"username": {"type": "string"},
                 "include_email": {"type": "boolean", "description": "optional; include email in the profile"}}
        req = ["username"]; name = "get_user_profile"
    return item(q, name, {"_desc": f"Query {name.replace('_',' ')}.", **props}, req, gold, f"f_opt_{i}")


FAMILIES = [
    (fam_rate, 2600), (fam_canonical, 3200), (fam_date, 1400),
    (fam_strnum, 1400), (fam_dict, 1600), (fam_list, 1400),
    (fam_plain, 4000), (fam_optional, 900),
]

SYSTEM = ("You are an expert in composing functions. You are given a question and a set of possible functions. "
          "Based on the question, you will need to make one or more function/tool calls to achieve the purpose. "
          "If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out. "
          "You should only return the function calls in your response and nothing else.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--bfcl-questions", default=None,
                    help="official v4 questions jsonl; collisions are dropped")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    real_qs = set()
    if args.bfcl_questions and os.path.exists(args.bfcl_questions):
        for l in open(args.bfcl_questions):
            if l.strip():
                real_qs.add(json.loads(l)["question"][0][0]["content"])

    items, manifest, dropped = [], {}, 0
    for fam, n in FAMILIES:
        made = 0
        for i in range(n):
            it = fam(rng, i)
            if it["question"] in real_qs:
                dropped += 1
                continue
            items.append(it)
            made += 1
        manifest[fam.__name__] = made

    rng.shuffle(items)
    n_val = max(50, int(len(items) * 0.04))
    val, train = items[:n_val], items[n_val:]

    os.makedirs(args.out_dir, exist_ok=True)
    # train/val are written in the RAW format consumed by the RL trainer
    # (methods/sphere_grpo.py load_rows: question/function/ground_truth).
    with open(os.path.join(args.out_dir, "train.jsonl"), "w") as f:
        for it in train:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "val.jsonl"), "w") as f:
        for it in val:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "items_bfcl_format.jsonl"), "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    manifest["total"] = len(items)
    manifest["dropped_vs_real"] = dropped
    json.dump(manifest, open(os.path.join(args.out_dir, "manifest.json"), "w"), indent=1)
    print(f"train={len(train)} val={len(val)} total={len(items)} dropped_vs_real={dropped}")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
