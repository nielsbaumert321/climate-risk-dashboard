import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import wbgapi as wb
from dash import Dash, dcc, html, Input, Output, callback

DATA_FILE = "public_emdat_custom_request_2026-06-22_All_countries.xlsx"

# ── Load EM-DAT ───────────────────────────────────────────────────────────────
_raw = pd.read_excel(DATA_FILE)

# Restrict to the five tracked hazard types and the 2000-2025 window so that
# every view (map, bars, normalizers) counts the same universe of events.
KNOWN_TYPES = {"Drought", "Extreme temperature", "Flood", "Storm", "Wildfire"}
df = _raw[
    _raw["Disaster Type"].isin(KNOWN_TYPES) &
    _raw["Start Year"].between(2000, 2024)
].copy()

country_agg = df.groupby(["ISO", "Country", "Region"]).agg(
    Event_Count=("DisNo.", "count"),
).reset_index()

# ── Load World Bank GDP + population ─────────────────────────────────────────
def _wb_long(indicator, value_col):
    raw = wb.data.DataFrame(indicator, time=range(2000, 2025), labels=False).reset_index()
    long = raw.melt(id_vars="economy", var_name="Year", value_name=value_col)
    long["Year"] = pd.to_numeric(
        long["Year"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce"
    ).astype("Int64")
    return long.rename(columns={"economy": "ISO"}).dropna(subset=["Year", value_col])

WB_CACHE = "wb_cache.csv"
if os.path.exists(WB_CACHE):
    wb_data = pd.read_csv(WB_CACHE)
    print(f"WB cache loaded: {len(wb_data):,} rows")
else:
    try:
        _gdp = _wb_long("NY.GDP.MKTP.KD", "GDP_USD")
        _pop = _wb_long("SP.POP.TOTL",    "Population")
        wb_data = _gdp.merge(_pop, on=["ISO", "Year"])
        wb_data.to_csv(WB_CACHE, index=False)
        print(f"WB data fetched and cached: {len(wb_data):,} rows")
    except Exception as e:
        wb_data = pd.DataFrame(columns=["ISO", "Year", "GDP_USD", "Population"])
        print(f"World Bank unavailable ({e}) — charts 2 & 3 show absolute values")

# ── Constants ─────────────────────────────────────────────────────────────────
REGION_SCOPE = {
    "World":    "world",
    "Africa":   "africa",
    "Asia":     "asia",
    "Americas": "world",
    "Europe":   "europe",
    "Oceania":  "world",
}

PERIOD_BINS    = [2000, 2005, 2010, 2015, 2020, 2025]
PERIOD_LABELS  = ["2000–2004", "2005–2009", "2010–2014", "2015–2019", "2020–2024"]
PERIOD_LENGTHS = {
    "2000–2004": 5, "2005–2009": 5, "2010–2014": 5,
    "2015–2019": 5, "2020–2024": 5,
}

_palette = px.colors.qualitative.Set2
TYPE_COLOR_MAP = {
    "Drought":             _palette[0],
    "Extreme temperature": _palette[1],
    "Flood":               _palette[2],
    "Storm":               _palette[3],
    "Wildfire":            _palette[4],
}
HAZARD_ORDER = sorted(TYPE_COLOR_MAP.keys())
REGIONS      = ["World"] + sorted(df["Region"].dropna().unique().tolist())

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_normalizers(sub):
    """Returns (period_gdp_5yr_sum, period_pop_5yr_sum) for countries in sub."""
    if wb_data.empty:
        return None, None
    isos   = sub["ISO"].dropna().unique()
    wb_sub = wb_data[wb_data["ISO"].isin(isos) & wb_data["Year"].between(2000, 2024)].copy()
    if wb_sub.empty:
        return None, None
    wb_sub["Period"] = pd.cut(
        wb_sub["Year"].astype(int), bins=PERIOD_BINS, labels=PERIOD_LABELS, right=False
    )
    period_gdp = (wb_sub.groupby("Period", observed=True)["GDP_USD"].sum()
                  .reindex(PERIOD_LABELS).replace(0, float("nan")))
    period_pop = (wb_sub.groupby("Period", observed=True)["Population"].sum()
                  .reindex(PERIOD_LABELS).replace(0, float("nan")))
    return period_gdp, period_pop


def _stacked_chart(data, title, ytitle, fmt=".1f", divisor=5):
    """Stacked bar chart — data is Period × HazardType; totals shown on top.
    divisor may be a scalar or a Series indexed by PERIOD_LABELS."""
    avg = data.div(divisor, axis=0) if hasattr(divisor, "__len__") else data / divisor
    totals = avg.sum(axis=1)
    ymax   = max(float(totals.max()) * 1.18, 1e-9)

    fig = go.Figure()
    for dtype in HAZARD_ORDER:
        if dtype not in avg.columns:
            continue
        fig.add_trace(go.Bar(
            name=dtype,
            x=avg.index.astype(str),
            y=avg[dtype].round(6),
            marker_color=TYPE_COLOR_MAP[dtype],
            hovertemplate=f"%{{y:{fmt}}}<extra>{dtype}</extra>",
        ))
    fig.add_trace(go.Scatter(
        x=totals.index.astype(str),
        y=totals.values,
        text=[f"{v:{fmt}}" for v in totals.values],
        mode="text",
        textposition="top center",
        textfont=dict(size=10, color="#333"),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.update_layout(
        barmode="stack",
        title=title,
        yaxis=dict(title=ytitle, range=[0, ymax]),
        height=370,
        legend=dict(orientation="h", y=-0.28),
        margin=dict(l=40, r=40, t=60, b=110),
        bargap=0.3,
    )
    return fig


def build_detail_charts(label, sub):
    sub2 = sub[sub["Start Year"].between(2000, 2024)].copy()
    sub2["Period"] = pd.cut(
        sub2["Start Year"], bins=PERIOD_BINS, labels=PERIOD_LABELS, right=False
    )
    g = sub2.groupby(["Period", "Disaster Type"], observed=True)

    pivot_events   = g.size().unstack(fill_value=0)
    pivot_damage   = g["Total Damage, Adjusted ('000 US$)"].sum().unstack(fill_value=0)
    pivot_deaths   = g["Total Deaths"].sum().unstack(fill_value=0)
    pivot_affected = g["Total Affected"].sum().unstack(fill_value=0)

    period_gdp, period_pop = get_normalizers(sub)

    yr_divisor = pd.Series(PERIOD_LENGTHS).reindex(PERIOD_LABELS)

    # Chart 1 — avg number of hazards / yr
    f1 = _stacked_chart(
        pivot_events,
        f"<b>{label}</b> — Avg number of hazards / yr",
        "Events / yr", fmt=".1f", divisor=yr_divisor,
    )

    # Chart 2 — economic damage as % of GDP (or absolute if no WB data)
    if period_gdp is not None:
        # (5yr_damage_000USD * 1000) / 5yr_GDP_USD * 100 = annual avg % GDP
        pivot_dmg_norm = pivot_damage.div(period_gdp, axis=0).fillna(0) * 1e5
        f2 = _stacked_chart(
            pivot_dmg_norm,
            f"<b>{label}</b> — Avg economic damage / yr (% of GDP)",
            "% of GDP / yr", fmt=".4f", divisor=1,
        )
    else:
        f2 = _stacked_chart(
            pivot_damage,
            f"<b>{label}</b> — Avg adj. economic damage / yr (000 USD)",
            "000 USD / yr", fmt=",.0f", divisor=yr_divisor,
        )

    # Chart 3 — deaths per million / yr (or absolute if no WB data)
    if period_pop is not None:
        # 5yr_deaths / 5yr_population * 1e6 = annual avg deaths per million
        pivot_deaths_norm = pivot_deaths.div(period_pop, axis=0).fillna(0) * 1e6
        f3 = _stacked_chart(
            pivot_deaths_norm,
            f"<b>{label}</b> — Avg deaths / million / yr",
            "Deaths / million / yr", fmt=".3f", divisor=1,
        )
    else:
        f3 = _stacked_chart(
            pivot_deaths,
            f"<b>{label}</b> — Avg deaths / yr",
            "Deaths / yr", fmt=",.0f", divisor=yr_divisor,
        )

    # Chart 4 — avg total affected / yr
    f4 = _stacked_chart(
        pivot_affected,
        f"<b>{label}</b> — Avg total affected / yr",
        "Persons / yr", fmt=",.0f", divisor=yr_divisor,
    )
    return f1, f2, f3, f4


# ── App ───────────────────────────────────────────────────────────────────────
app = Dash(__name__)
server = app.server  # exposed for gunicorn: gunicorn app:server

app.layout = html.Div([
    html.H2("EM-DAT Interactive Disaster Map", style={"marginBottom": "4px"}),
    html.P("Drill down: World → Region → Country", style={"color": "#666", "marginTop": 0}),

    html.Div([
        html.Div([
            html.Label("Region"),
            dcc.Dropdown(
                id="region-dd",
                options=[{"label": r, "value": r} for r in REGIONS],
                value="World",
                clearable=False,
            ),
        ], style={"width": "180px"}),

        html.Div([
            html.Label("Country"),
            dcc.Dropdown(id="country-dd", value="(all)", clearable=False),
        ], style={"width": "240px"}),
    ], style={"display": "flex", "gap": "16px", "alignItems": "flex-end", "marginBottom": "12px"}),

    dcc.Graph(id="map", style={"height": "520px"}),
    html.Div(id="detail-charts"),
], style={"fontFamily": "sans-serif", "maxWidth": "1400px", "margin": "24px auto", "padding": "0 16px"})


@callback(
    Output("country-dd", "options"),
    Output("country-dd", "value"),
    Input("region-dd", "value"),
)
def update_country_options(region):
    if region == "World":
        countries = sorted(df["Country"].dropna().unique().tolist())
    else:
        countries = sorted(df[df["Region"] == region]["Country"].dropna().unique().tolist())
    options = [{"label": "(all)", "value": "(all)"}] + [{"label": c, "value": c} for c in countries]
    return options, "(all)"


@callback(
    Output("map", "figure"),
    Input("region-dd", "value"),
    Input("country-dd", "value"),
)
def update_map(region, country):
    if country and country != "(all)":
        data      = country_agg[country_agg["Country"] == country]
        scope     = "world"
        fitbounds = "locations"
        title_loc = country
    else:
        data      = country_agg if region == "World" else country_agg[country_agg["Region"] == region]
        scope     = REGION_SCOPE.get(region, "world")
        fitbounds = "locations" if region != "World" else False
        title_loc = region

    fig = px.choropleth(
        data,
        locations="ISO",
        color="Event_Count",
        hover_name="Country",
        hover_data={"ISO": False, "Event_Count": ":,"},
        color_continuous_scale="Reds",
        title=f"<b>Number of disaster events</b> — {title_loc}",
        scope=scope,
        fitbounds=fitbounds,
        basemap_visible=True,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar=dict(title="Events"),
    )
    return fig


@callback(
    Output("detail-charts", "children"),
    Input("country-dd", "value"),
    Input("region-dd", "value"),
)
def update_detail(country, region):
    if country and country != "(all)":
        sub, label = df[df["Country"] == country], country
    elif region and region != "World":
        sub, label = df[df["Region"] == region], region
    else:
        sub, label = df, "Global"

    f1, f2, f3, f4 = build_detail_charts(label, sub)

    row = lambda a, b: html.Div(
        [dcc.Graph(figure=a, style={"flex": "1"}),
         dcc.Graph(figure=b, style={"flex": "1"})],
        style={"display": "flex", "gap": "16px"},
    )

    return html.Div([
        row(f1, f2),
        row(f3, f4),
    ], style={"marginTop": "16px", "display": "flex", "flexDirection": "column", "gap": "16px"})


if __name__ == "__main__":
    app.run(debug=False)
