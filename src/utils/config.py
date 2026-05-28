import os

PROJECT_NAME = "World Marathon Majors Run Analytics Challenge"
PROJECT_VERSION = "1.0.0"

MARATHON_DISTANCE_KM = 42.195
MARATHON_DISTANCE_MI = 26.2188

DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
EXTERNAL_DIR = os.path.join(DATA_DIR, "external")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")

MODELS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
TRAINED_MODELS_DIR = os.path.join(MODELS_DIR, "trained")
METRICS_DIR = os.path.join(MODELS_DIR, "metrics")

REPORTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "reports"))
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

YEARS = list(range(2018, 2026))

RACE_NAMES = [
    "Tokyo",
    "Boston",
    "London",
    "Berlin",
    "Chicago",
    "New York City",
]

RACE_CITIES = ["Tokyo", "Boston", "London", "Berlin", "Chicago", "New York"]

RACE_COUNTRIES = [
    "Japan",
    "United States",
    "United Kingdom",
    "Germany",
    "United States",
    "United States",
]

RACE_MONTHS = {
    "Tokyo": 3,
    "Boston": 4,
    "London": 4,
    "Berlin": 9,
    "Chicago": 10,
    "New York City": 11,
}

CANCELLED_RACES = {
    (2020, "Boston"),
    (2020, "Berlin"),
    (2020, "Chicago"),
    (2020, "New York City"),
}

PERFORMANCE_CATEGORIES = ["Elite", "Advanced", "Intermediate", "Recreational"]

AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]

COLOR_PALETTE = {
    "background": "#F8FAFC",
    "primary": "#2563EB",
    "secondary": "#14B8A6",
    "accent": "#F97316",
    "success": "#22C55E",
    "warning": "#F59E0B",
    "text": "#0F172A",
    "muted": "#64748B",
    "card_bg": "#FFFFFF",
    "border": "#E2E8F0",
}

PLOTLY_TEMPLATE = "plotly_white"

DATA_SOURCES = {
    "world_marathon_majors": {
        "name": "World Marathon Majors Dataset",
        "url": "https://www.kaggle.com/datasets/emmanuelfuentes/world-marathon-majors",
        "description": "Comprehensive results from all six World Marathon Majors covering finisher times, demographics, and splits.",
        "coverage": "2018-2024",
    },
    "boston_marathon": {
        "name": "Boston Marathon Results",
        "url": "https://www.kaggle.com/datasets/rojour/boston-marathon-results",
        "description": "Detailed Boston Marathon finisher data including split times, demographics, and qualifying status.",
        "coverage": "2015-2023",
    },
    "berlin_marathon": {
        "name": "Berlin Marathon Results",
        "url": "https://www.kaggle.com/datasets/runningwithworld/berlin-marathon",
        "description": "Berlin Marathon results with finisher times, nationalities, age groups, and gender divisions.",
        "coverage": "2010-2023",
    },
    "chicago_marathon": {
        "name": "Chicago Marathon Results",
        "url": "https://www.kaggle.com/datasets/mexwell/chicago-marathon-results",
        "description": "Chicago Marathon historical results with split times, pace data, and runner demographics.",
        "coverage": "2014-2023",
    },
    "london_marathon": {
        "name": "London Marathon Results",
        "url": "https://www.kaggle.com/datasets/thedevastator/london-marathon-results",
        "description": "London Marathon finisher data including finish times, club affiliations, and age category results.",
        "coverage": "2010-2023",
    },
    "new_york_marathon": {
        "name": "New York City Marathon Results",
        "url": "https://www.kaggle.com/datasets runningwithworld/new-york-city-marathon",
        "description": "NYC Marathon results with finisher times, borough splits, and runner demographics.",
        "coverage": "2015-2023",
    },
    "tokyo_marathon": {
        "name": "Tokyo Marathon Results",
        "url": "https://www.kaggle.com/datasets/runningwithworld/tokyo-marathon",
        "description": "Tokyo Marathon results including finisher times, nationalities, and age group breakdowns.",
        "coverage": "2017-2024",
    },
    "marathon_winners": {
        "name": "Marathon World Records and Winners",
        "url": "https://www.kaggle.com/datasets/josephvm/marathon-world-records",
        "description": "Historical winners and record-setting performances from major marathons worldwide.",
        "coverage": "2000-2024",
    },
}
