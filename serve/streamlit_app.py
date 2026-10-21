"""Premium explorer: move driver age, vehicle age and bonus-malus; NN vs GLM side by side.

    streamlit run serve/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from serve.predictor import Predictor  # noqa: E402
from src.data.schema import AREAS, REGIONS, VEH_BRANDS  # noqa: E402

st.set_page_config(page_title="Premium explorer", layout="wide")


@st.cache_resource
def predictor() -> Predictor:
    return Predictor()


p = predictor()
st.title("Pure premium explorer: neural model vs Poisson/Gamma GLM")

with st.sidebar:
    st.header("Policy")
    drivage = st.slider("Driver age", 18, 90, 40)
    vehage = st.slider("Vehicle age", 0, 20, 5)
    bm = st.slider("Bonus-malus", 50, 150, 50)
    vehpower = st.slider("Vehicle power", 4, 15, 6)
    density = st.number_input("Population density (per km²)", 1, 27000, 1200)
    area = st.selectbox("Area", AREAS, index=2)
    brand = st.selectbox("Vehicle brand", VEH_BRANDS, index=VEH_BRANDS.index("B12"))
    gas = st.selectbox("Fuel", ["Regular", "Diesel"])
    region = st.selectbox("Region", REGIONS, index=REGIONS.index("Ile-de-France"))

policy = dict(Area=area, VehPower=vehpower, VehAge=vehage, DrivAge=drivage, BonusMalus=bm,
              VehBrand=brand, VehGas=gas, Density=density, Region=region, Exposure=1.0)
nn, glm = p.quote(policy, "nn"), p.quote(policy, "glm")

c1, c2, c3 = st.columns(3)
c1.metric("Pure premium: NN", f"€{nn['pure_premium']:.0f}", f"{nn['pure_premium'] - glm['pure_premium']:+.0f} vs GLM")
c2.metric("Pure premium: GLM", f"€{glm['pure_premium']:.0f}")
c3.metric("Risk segment / anomaly", f"#{nn['risk_segment']}", "unusual profile" if nn["anomaly"] else "typical profile",
          delta_color="inverse" if nn["anomaly"] else "off")
st.table(pd.DataFrame({"NN": nn, "GLM": glm}).loc[["expected_frequency", "expected_severity", "pure_premium"]].T.round(4))

st.subheader("How the premium moves with each slider (other inputs fixed)")
cols = st.columns(3)
for col, (feat, grid) in zip(cols, [("DrivAge", np.arange(18, 91)), ("VehAge", np.arange(0, 21)),
                                    ("BonusMalus", np.arange(50, 151, 2))]):
    df = pd.DataFrame([policy] * len(grid)).assign(**{feat: grid})
    curve = pd.DataFrame({feat: grid, "NN": p.quote_frame(df, "nn").pure_premium.values,
                          "GLM": p.quote_frame(df, "glm").pure_premium.values}).set_index(feat)
    col.markdown(f"**{feat}**")
    col.line_chart(curve)
