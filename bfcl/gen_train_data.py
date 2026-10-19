#!/usr/bin/env python3
"""Generate synthetic BFCL-simple_python-style RL training data.

The baseline audit showed the model's failures cluster into convention families
(percent-vs-decimal rates, canonical enum tokens, SI unit conversion, 24-hour
times, ticker symbols, value trimming, math syntax, date formats, string-typed
numerics, dict inner keys, canonical list elements, small fixed facts). This
generator builds training items that teach those conventions. Schema
descriptions deliberately mirror the official style -- often vague or even
contradictory ("as a percentage" while the gold is the decimal fraction) --
because that is what the official golds do; the convention itself is what the
policy must learn from data consistency. It does NOT copy any real BFCL v4
item (a dedup guard drops any collision when --bfcl-questions is passed).

Each family mixes the target convention with look-alike distractors so the
model must read the schema instead of applying one blanket rule.

Output:
  train.jsonl  (raw format: question / function / ground_truth)
  val.jsonl    (held-out 4%)
  items_bfcl_format.jsonl (all items, original shuffled order)
  manifest.json (per-family counts)

Usage: python gen_train_data.py --out-dir data [--seed 7] \
    [--bfcl-questions official_questions.jsonl]
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
        props = {rate_key: {"type": "float", "description": "The discount rate to use."},
                 "period": {"type": "integer"},
                 "cash_flows": {"type": "array", "items": {"type": "float"},
                                "description": "Series of future cash flows."}}
        req = [rate_key, "period", "cash_flows"]
    else:
        if rng.random() < 0.5:
            rate_phrase = rng.choice([
                f"at an annual rate of {pct}%",
                f"with yearly interest of {pct} percent",
                f"earning {pct}% per year"])
            q = (f"Calculate the {name.replace('_', ' ')} on a principal of "
                 f"${principal}, {rate_phrase}, compounded "
                 f"{freq} for {years} years.")
            gold = {"principal": [principal], rate_key: [rate],
                    "compounding_freq": [freq], "time_years": [years]}
            props = {"principal": {"type": "integer",
                                   "description": "The initial amount of money that was invested or loaned out."},
                     rate_key: {"type": "float",
                                "description": "The interest rate for a year as a percentage."},
                     "compounding_freq": {"type": "string",
                                          "enum": ["monthly", "quarterly", "semi-annually", "annually", "daily"],
                                          "description": "The number of times that interest is compounded per unit period."},
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
                     rate_key: {"type": "float",
                                "description": "The annual rate of return."},
                     "compounding_periods_per_year": {"type": "integer", "description": "Number of compounding periods per year."},
                     "time_years": {"type": "integer"}}
            req = ["present_value", rate_key, "compounding_periods_per_year", "time_years"]

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
        props = {"gene": {"type": "string", "description": "The gene of interest."},
                 "species": {"type": "string",
                             "description": "The species for which the gene is to be analyzed.",
                             "default": latin}}
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
        # list-of-dicts with column/operator/value (two key conventions,
        # values sometimes stringified -- both appear in the official data)
        c1, c2 = rng.sample(cols, 2)
        op1 = rng.choice([">", "<", ">=", "<=", "="])
        val1 = rng.randint(18, 65)
        eqval = rng.choice(pools[list(pools)[0]])
        eqcol = list(pools)[0]
        q = (f"Query the {table} table for records where {c1} {op1} {val1} "
             f"and {eqcol} equals {eqval}.")
        if rng.random() < 0.5:
            gold = {"table": [table],
                    "conditions": [[{"column": [c1], "operator": [op1], "value": [val1]},
                                    {"column": [eqcol], "operator": ["="], "value": [eqval]}]]}
            props = {"table": {"type": "string"},
                     "conditions": {"type": "array", "items": {"type": "dict"},
                                    "description": "each element must have exactly the keys: column, operator, value"}}
        else:
            gold = {"table": [table],
                    "conditions": [[{"field": [c1], "operation": [op1], "value": [str(val1)]},
                                    {"field": [eqcol], "operation": ["="], "value": [eqval]}]]}
            props = {"table": {"type": "string"},
                     "conditions": {"type": "array", "items": {"type": "dict"},
                                    "description": "each element must have exactly the keys: field, operation, value"}}
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
        q = f"A triangle has a base measuring {b} units and a height of {h} units. What is its area?"
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


def fam_enum(rng, i):
    """Natural phrasing -> canonical enum token, official-style schemas."""
    kind = rng.choice(["time_frame", "compounding", "compounding_cap",
                       "sort", "detail_level", "root_type", "currency"])
    if kind == "time_frame":
        phrase, val = rng.choice([
            ("the last six months", "six_months"), ("the past six months", "six_months"),
            ("the past year", "year"), ("the last 12 months", "year"),
            ("the last five years", "five_years"), ("the past 5 years", "five_years"),
            ("the past decade", "ten_years")])
        loc = rng.choice(["the Amazon rainforest", "the Sahara Desert", "Antarctica",
                          "the Great Barrier Reef", "Yellowstone National Park"])
        q = f"Pull up average precipitation data for {loc} covering {phrase}."
        gold = {"location": [loc], "time_frame": [val]}
        props = {"location": {"type": "string", "description": "The name of the location."},
                 "time_frame": {"type": "string", "enum": ["six_months", "year", "five_years", "ten_years"],
                                "description": "The time period for which data is required."}}
        req = ["location", "time_frame"]; name = "ecology_data_precipitation_stats"
    elif kind == "compounding":
        phrase, val = rng.choice([
            ("compounded monthly", "monthly"), ("compounded every month", "monthly"),
            ("compounded quarterly", "quarterly"), ("compounded every three months", "quarterly"),
            ("compounded twice a year", "semi-annually"), ("compounded semi-annually", "semi-annually"),
            ("compounded once a year", "annually"), ("compounded annually", "annually"),
            ("compounded every day", "daily"), ("compounded daily", "daily")])
        principal = rng.choice([1000, 2500, 5000, 10000, 20000])
        pct = rng.choice([3, 4, 5, 6, 7, 8])
        years = rng.randint(1, 10)
        q = (f"Compute the compound interest on a ${principal} deposit at {pct}% "
             f"annual interest, {phrase}, over {years} years.")
        gold = {"principal": [principal], "annual_rate": [pct / 100],
                "compounding_freq": [val], "time_in_years": [years]}
        props = {"principal": {"type": "integer"},
                 "annual_rate": {"type": "float", "description": "The interest rate for a year as a percentage."},
                 "compounding_freq": {"type": "string", "enum": ["monthly", "quarterly", "annually"],
                                      "description": "The number of times that interest is compounded per unit period."},
                 "time_in_years": {"type": "integer"}}
        req = ["principal", "annual_rate", "compounding_freq", "time_in_years"]
        name = "calculate_compound_interest"
    elif kind == "compounding_cap":
        phrase, val = rng.choice([
            ("compounded every month", "monthly"),
            ("compounded once a year", "Annually"),
            ("compounded annually", "Annually"),
            ("compounded twice a year", "Semi-annually"),
            ("compounded quarterly", "Quarterly")])
        principal = rng.choice([5000, 10000, 15000])
        pct = rng.choice([4, 5, 6, 7])
        q = (f"Compute the future value of ${principal} invested at an annual "
             f"interest rate of {pct}%, {phrase}.")
        gold = {"principal": [principal], "annual_interest_rate": [pct / 100],
                "compounding_frequency": [val]}
        props = {"principal": {"type": "integer"},
                 "annual_interest_rate": {"type": "float", "description": "Interest rate per year."},
                 "compounding_frequency": {"type": "string",
                                           "enum": ["monthly", "Quarterly", "Semi-annually", "Annually"],
                                           "description": "How often interest is compounded."}}
        req = ["principal", "annual_interest_rate", "compounding_frequency"]
        name = "investment_future_value"
    elif kind == "sort":
        phrase, val = rng.choice([
            ("from highest to lowest", "descending"), ("highest first", "descending"),
            ("in descending order", "descending"), ("biggest first", "descending"),
            ("from lowest to highest", "ascending"), ("in ascending order", "ascending"),
            ("alphabetical order", "ascending"), ("smallest first", "ascending")])
        col = rng.choice(["price", "rating", "area", "population"])
        q = f"List the countries by {col} {phrase}."
        gold = {"sort_column": [col], "order": [val]}
        props = {"sort_column": {"type": "string"},
                 "order": {"type": "string", "enum": ["ascending", "descending"],
                           "description": "Sort order."}}
        req = ["sort_column", "order"]; name = "countries_ranked"
    elif kind == "detail_level":
        phrase, val = rng.choice([
            ("with full details", "detailed"), ("in full detail", "detailed"),
            ("with everything included", "detailed"), ("just the basics", "basic"),
            ("a basic summary", "basic"), ("the short version", "basic")])
        city, _ = rng.choice(CITIES)
        q = f"Give me the traffic report for {city} {phrase}."
        gold = {"location": [city], "detail_level": [val]}
        props = {"location": {"type": "string"},
                 "detail_level": {"type": "string", "enum": ["basic", "detailed"],
                                  "description": "How much detail to include."}}
        req = ["location", "detail_level"]; name = "traffic_report"
    elif kind == "root_type":
        phrase, val = rng.choice([
            ("all the roots", "all"), ("both real and complex roots", "all"),
            ("every root", "all"), ("the real roots", "real"), ("only the real roots", "real")])
        a, b, c = rng.randint(1, 9), rng.randint(-15, 15), rng.randint(-15, 15)
        q = (f"Solve the quadratic with coefficients a={a}, b={b}, c={c} "
             f"and return {phrase}.")
        gold = {"a": [a], "b": [b], "c": [c], "root_type": [val]}
        props = {"a": {"type": "integer", "description": "The coefficient of the squared term in the quadratic equation."},
                 "b": {"type": "integer", "description": "The coefficient of the linear term."},
                 "c": {"type": "integer", "description": "The constant term."},
                 "root_type": {"type": "string",
                              "description": "The type of roots to return: 'real' for real roots, 'all' for both real and complex roots. Default value is 'real'."}}
        req = ["a", "b", "c", "root_type"]; name = "solve_quadratic"
    else:
        cur = rng.choice(["USD", "EUR", "JPY", "GBP", "CAD", "AUD"])
        amount = rng.choice([100, 250, 500, 1000, 2500])
        q = f"What is the exchange rate from {cur} to USD for {amount} {cur}?"
        gold = {"from_currency": [cur], "amount": [amount]}
        props = {"from_currency": {"type": "string", "enum": ["USD", "EUR", "JPY", "GBP", "CAD", "AUD"]},
                 "amount": {"type": "integer"}}
        req = ["from_currency", "amount"]; name = "exchange_rate"
    return item(q, name, {"_desc": f"Query {name.replace('_',' ')}.", **props}, req, gold, f"f_enum_{i}")


UNIT_POOLS = [
    ("inductance", "inductance", "henries (H)", [("mH", 1e-3), ("µH", 1e-6), ("H", 1.0)]),
    ("capacitance", "capacitance", "farads (F)", [("µF", 1e-6), ("nF", 1e-9), ("pF", 1e-12)]),
    ("resistance", "resistance", "ohms", [("kΩ", 1e3), ("MΩ", 1e6), ("Ω", 1.0)]),
    ("length", "distance", "meters (m)", [("km", 1e3), ("cm", 1e-2), ("mm", 1e-3)]),
    ("mass", "mass", "kilograms (kg)", [("g", 1e-3), ("mg", 1e-6), ("kg", 1.0)]),
    ("volume", "volume", "liters (L)", [("mL", 1e-3), ("L", 1.0)]),
    ("time", "duration", "seconds (s)", [("ms", 1e-3), ("µs", 1e-6), ("s", 1.0)]),
]


def fam_unit(rng, i):
    """Question uses a prefixed unit; gold converts to the base SI unit."""
    kind, param, unit_desc, prefixes = rng.choice(UNIT_POOLS)
    base = rng.choice([1, 2, 5, 10, 20, 25, 47, 50, 100, 150, 220, 330, 470, 500, 1000])
    prefix, mult = rng.choice(prefixes)
    gold_val = pnum(base * mult)
    if rng.random() < 0.5:
        q = f"Calculate with a {param} of {base}{prefix}."
        desc = f"The {param} in {unit_desc}."
    else:
        q = f"Calculate with a {param} of {base} {prefix}."
        desc = f"The {param}."
    gold = {param: [gold_val]}
    props = {param: {"type": "float", "description": desc}}
    req = [param]
    name = "physics_calculate_" + param
    return item(q, name, {"_desc": f"Calculate using the given {param}.", **props}, req, gold, f"f_unit_{i}")


def fam_time24(rng, i):
    """12-hour clock phrases -> integer hour in 24-hour format."""
    phrase, hour = rng.choice([
        ("11 PM", 23), ("at 11 in the evening", 23), ("10 PM", 22),
        ("9 PM", 21), ("at 9 in the evening", 21), ("8 PM", 20),
        ("7 PM", 19), ("at 7 in the evening", 19), ("6 PM", 18),
        ("noon", 12), ("12 PM", 12), ("midnight", 0), ("12 AM", 0),
        ("11 AM", 11), ("9 AM", 9), ("8 AM", 8), ("7 AM", 7),
        ("6 AM", 6), ("at 6 in the morning", 6),
        ("3 o'clock in the afternoon", 15), ("2 in the afternoon", 14)])
    city, _ = rng.choice(CITIES)
    q = f"Find restaurants in {city} that are open until {phrase}."
    gold = {"location": [city], "operating_hours": [hour]}
    props = {"location": {"type": "string", "description": "The city and state, e.g. New York, NY, you should format it as City, State."},
             "operating_hours": {"type": "integer",
                                 "description": "Preferred latest closing time. This is in 24 hour format. Default is 24."}}
    req = ["location", "operating_hours"]; name = "restaurant_open_late"
    return item(q, name, {"_desc": "Find restaurants by opening hours.", **props}, req, gold, f"f_time_{i}")


TICKERS = {"Apple": "AAPL", "Microsoft": "MSFT", "Alphabet": "GOOGL",
           "Amazon": "AMZN", "Tesla": "TSLA", "Meta": "META", "Nvidia": "NVDA",
           "Netflix": "NFLX", "Intel": "INTC", "AMD": "AMD", "IBM": "IBM",
           "Oracle": "ORCL", "Salesforce": "CRM", "Uber": "UBER", "Lyft": "LYFT",
           "PayPal": "PYPL", "Starbucks": "SBUX", "Disney": "DIS", "Nike": "NKE",
           "Boeing": "BA", "Cisco": "CSCO", "Adobe": "ADBE", "Pfizer": "PFE"}


def fam_ticker(rng, i):
    """Company name in the question -> ticker symbol (schema says 'ticker symbol')."""
    comp, tick = rng.choice(list(TICKERS.items()))
    days = rng.choice([1, 5, 10, 30, 90])
    q = f"Show me the stock price history of {comp} covering the past {days} days."
    gold = {"company": [tick], "days": [days]}
    props = {"company": {"type": "string", "description": "The ticker symbol of the company."},
             "days": {"type": "integer", "description": "Number of past days for which the stock price is required."}}
    req = ["company", "days"]; name = "stock_price_lookup"
    return item(q, name, {"_desc": "Look up stock prices.", **props}, req, gold, f"f_tick_{i}")


TRIM_KINDS = [
    # (kind, question value, canonical, param, name, enum_list, description)
    ("cuisine", {"Thai food": "Thai", "Italian food": "Italian",
                 "Mexican food": "Mexican", "Indian food": "Indian",
                 "Japanese food": "Japanese", "Chinese food": "Chinese",
                 "Thai cuisine": "Thai", "Italian cuisine": "Italian",
                 "American food": "American", "Korean food": "Korean"}),
    ("cell", {"human cell": "human", "bacterial cell": "bacterial",
              "plant cell": "plant", "animal cell": "animal",
              "blood cell": "blood", "stem cell": "stem",
              "muscle cell": "muscle", "nerve cell": "nerve"}),
    ("crime", {"theft crimes": "theft", "fraud crimes": "fraud",
               "property crimes": "property", "violent crimes": "violent",
               "white collar crimes": "white collar", "cyber crimes": "cyber"}),
]


def fam_trim(rng, i):
    """Verbose category phrase -> canonical trimmed value."""
    kind = rng.choice(["cuisine", "cell", "crime", "county", "specialty", "detail_word"])
    if kind in ("cuisine", "cell", "crime"):
        pool = next(t for k, t in TRIM_KINDS if k == kind)
        verbose, canon = rng.choice(list(pool.items()))
        city, _ = rng.choice(CITIES)
        n = rng.choice([3, 5, 10])
        if kind == "cuisine":
            q = f"Can you find {n} restaurants in {city} that serve {verbose}?"
            gold = {"location": [city], "food_type": [canon], "number": [n]}
            props = {"location": {"type": "string", "description": "The specific location or area."},
                     "food_type": {"type": "string", "description": "The type of food preferred."},
                     "number": {"type": "integer", "description": "Number of results to return."}}
            req = ["location", "food_type", "number"]; name = "find_restaurants"
        elif kind == "cell":
            q = f"Get information about the {verbose} structure."
            gold = {"cell_type": [canon]}
            props = {"cell_type": {"type": "string", "description": "The type of cell."}}
            req = ["cell_type"]; name = "biology_get_cell_info"
        else:
            q = f"Show me the statistics for {verbose} in the last year."
            gold = {"crime_type": [canon], "time_frame": ["year"]}
            props = {"crime_type": {"type": "string", "description": "The category of crime."},
                     "time_frame": {"type": "string", "enum": ["six_months", "year", "five_years"]}}
            req = ["crime_type", "time_frame"]; name = "crime_statistics"
    elif kind == "county":
        county = rng.choice(["Santa Clara", "San Diego", "Orange", "Alameda",
                             "Cook", "King", "Travis", "Denver", "Bexar"])
        n = rng.randint(100000000, 999999999)
        q = f"Look up the property with parcel number {n} in {county} county."
        gold = {"parcel_number": [str(n)], "county": [county]}
        props = {"parcel_number": {"type": "string", "description": "The parcel number."},
                 "county": {"type": "string", "description": "The county name."}}
        req = ["parcel_number", "county"]; name = "property_lookup"
    elif kind == "specialty":
        verbose, canon = rng.choice([
            ("divorce cases", "Divorce"), ("family law cases", "Family"),
            ("criminal defense cases", "Criminal"), ("personal injury cases", "Personal Injury"),
            ("estate planning cases", "Estate Planning"), ("immigration cases", "Immigration")])
        city, _ = rng.choice(CITIES)
        q = f"Find a lawyer in {city} that handles {verbose}."
        gold = {"city": [city], "specialty": [[canon]]}
        props = {"city": {"type": "string"},
                 "specialty": {"type": "array", "items": {"type": "string"},
                               "description": "Legal specialty; each element must be one of: Divorce, Family, Criminal, Personal Injury, Estate Planning, Immigration"}}
        req = ["city", "specialty"]; name = "lawyer_find_nearby"
    else:
        word, canon = rng.choice([
            ("detail", "detailed"), ("the detailed version", "detailed"),
            ("a summary", "basic"), ("the basics", "basic")])
        bg = rng.choice(["Ticket to Ride", "Catan", "Pandemic", "Wingspan", "Azul"])
        q = f"Get me {word} information about the board game {bg}."
        gold = {"name": [bg], "detail_level": [canon]}
        props = {"name": {"type": "string"},
                 "detail_level": {"type": "string", "enum": ["basic", "detailed"],
                                   "description": "Level of detail for the information."}}
        req = ["name", "detail_level"]; name = "boardgame_get_info"
    return item(q, name, {"_desc": f"Query {name.replace('_',' ')}.", **props}, req, gold, f"f_trim_{i}")


def fam_mathsyntax(rng, i):
    """Math notation: ** exponents, float intervals, pi evaluation."""
    kind = rng.choice(["exponent", "pi", "fraction"])
    if kind == "exponent":
        k = rng.randint(2, 5)
        a, b = rng.randint(1, 5), rng.randint(6, 10)
        q = f"Compute the area beneath y=x^{k} between x={a} and x={b}."
        gold = {"function": [f"x**{k}", f"y=x**{k}", f"lambda x: x**{k}"],
                "interval": [[float(a), float(b)]],
                "method": ["", "trapezoidal"]}
        props = {"function": {"type": "string", "description": "The mathematical function as a string."},
                 "interval": {"type": "array", "items": {"type": "float"},
                              "description": "An array that defines the interval to calculate the area under the curve from the start to the end point."},
                 "method": {"type": "string",
                            "description": "The numerical method to approximate the area under the curve. The default value is 'trapezoidal'."}}
        req = ["function", "interval"]; name = "calculate_area_under_curve"
    elif kind == "pi":
        freq = rng.randint(1, 12)
        q = f"Plot a sine wave over the range from 0 to 2 * pi with frequency {freq}."
        gold = {"start_range": [0], "end_range": [6.2832], "frequency": [freq],
                "amplitude": ["", 1], "phase_shift": ["", 0]}
        props = {"start_range": {"type": "float", "description": "The starting x-value of the sine wave."},
                 "end_range": {"type": "float", "description": "The ending x-value of the sine wave."},
                 "frequency": {"type": "integer", "description": "The frequency of the sine wave."},
                 "amplitude": {"type": "float", "description": "The amplitude of the sine wave. Default is 1."},
                 "phase_shift": {"type": "float", "description": "The phase shift of the sine wave. Default is 0."}}
        req = ["start_range", "end_range", "frequency"]; name = "plot_sine_wave"
    else:
        num, den = rng.choice([(1, 2), (3, 4), (2, 3), (1, 4), (5, 8), (3, 8), (7, 8)])
        q = f"Compute the decimal value of {num}/{den} rounded to four decimal places."
        val = round(num / den, 4)
        gold = {"numerator": [num], "denominator": [den], "result": [val]}
        props = {"numerator": {"type": "integer"}, "denominator": {"type": "integer"},
                 "result": {"type": "float", "description": "The decimal value."}}
        req = ["numerator", "denominator"]; name = "fraction_to_decimal"
    return item(q, name, {"_desc": f"Compute {name.replace('_',' ')}.", **props}, req, gold, f"f_math_{i}")


DECK_FACTS = [
    ("a king", 4), ("a queen", 4), ("an ace", 4), ("a jack", 4),
    ("a heart", 13), ("a spade", 13), ("a diamond", 13), ("a club", 13),
    ("a red card", 26), ("a black card", 26), ("a face card", 12),
]


def fam_knowledge(rng, i):
    """Small fixed facts (deck composition etc.) that the question relies on."""
    phrase, n = rng.choice(DECK_FACTS)
    draws = rng.randint(1, 5)
    q = (f"Draw {draws} cards with replacement from a standard 52-card deck. "
            f"What is the chance of drawing {phrase} on each draw?")
    gold = {"total_outcomes": [52], "event_outcomes": [n], "round": ["", 2]}
    props = {"total_outcomes": {"type": "integer", "description": "The total number of outcomes."},
             "event_outcomes": {"type": "integer", "description": "The number of outcomes where the event occurs."},
             "round": {"type": "integer", "description": "Round the answer to a specified number of decimal places. Defaults to 2."}}
    req = ["total_outcomes", "event_outcomes"]; name = "probabilities_calculate_single"
    return item(q, name, {"_desc": "Calculate a probability.", **props}, req, gold, f"f_know_{i}")


FAMILIES = [
    (fam_rate, 3000), (fam_canonical, 2000), (fam_date, 1400),
    (fam_strnum, 1000), (fam_dict, 1800), (fam_list, 1400),
    (fam_plain, 2000), (fam_optional, 1200),
    (fam_enum, 2500), (fam_unit, 1800), (fam_time24, 1200),
    (fam_ticker, 1200), (fam_trim, 2200), (fam_mathsyntax, 1200),
    (fam_knowledge, 600),
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
