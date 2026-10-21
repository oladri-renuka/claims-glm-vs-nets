"""Column names and the known category values (shared by training and the API)."""

AREAS = ["A", "B", "C", "D", "E", "F"]
VEH_GAS = ["Diesel", "Regular"]
VEH_BRANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B10", "B11", "B12", "B13", "B14"]
REGIONS = [
    "Alsace", "Aquitaine", "Auvergne", "Basse-Normandie", "Bourgogne", "Bretagne", "Centre",
    "Champagne-Ardenne", "Corse", "Franche-Comte", "Haute-Normandie", "Ile-de-France",
    "Languedoc-Roussillon", "Limousin", "Midi-Pyrenees", "Nord-Pas-de-Calais", "Pays-de-la-Loire",
    "Picardie", "Poitou-Charentes", "Provence-Alpes-Cotes-D'Azur", "Rhone-Alpes",
]
CATEGORIES = {"Area": AREAS, "VehBrand": VEH_BRANDS, "Region": REGIONS}

# Raw rating factors (Exposure is never a feature: it enters only as an offset).
FEATURES = ["Area", "VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region"]
# Continuous inputs for the neural nets (after capping; Density is used on the log scale).
NN_CONTINUOUS = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "LogDensity", "VehGasDiesel"]
NN_CATEGORICAL = ["Area", "VehBrand", "Region"]

# Valid input ranges (raw, before capping) taken from the observed data.
RANGES = {
    "VehPower": (4, 15), "VehAge": (0, 100), "DrivAge": (18, 100),
    "BonusMalus": (50, 230), "Density": (1, 27000), "Exposure": (1e-4, 2.01),
}
