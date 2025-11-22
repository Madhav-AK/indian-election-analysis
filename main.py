import io, base64, textwrap
from typing import List
import plotly.io as pio
import plotly.express as px
pio.templates.default = "plotly_dark"
px.defaults.template = "plotly_dark"
import india_elections_figures
import india_election_maps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from dash import Dash, html, dcc
import dash_bootstrap_components as dbc


# Helpers
def slugify(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in s).strip("-")


def _enforce_dark(fig):
    """Force dark regardless of internal template."""
    try:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f0f12",
            plot_bgcolor="#0f0f12",
            font=dict(color="#e6e6e9"),
        )
    except Exception:
        pass
    return fig


def fig_card(title: str, fig, note: str = None):
    fig = _enforce_dark(fig)
    return dbc.Card(
        [
            dbc.CardHeader(title, className="card-title-dark"),
            dbc.CardBody(
                [
                    dcc.Graph(
                        id=slugify(title),
                        figure=fig,
                        config={"displayModeBar": "hover", "toImageButtonOptions": {"scale": 2}},
                        style={"height": "480px"}
                    ),
                    html.Small(note, className="text-muted") if note else None,
                ]
            ),
        ],
        className="shadow-soft bg-dark-2 text-light h-100",
    )


def img_card(title: str, img_src: str, caption: str = None):
    return dbc.Card(
        [
            dbc.CardHeader(title, className="card-title-dark"),
            dbc.CardBody(
                [
                    html.Img(src=img_src, style={"width": "100%", "borderRadius": "10px"}),
                    html.Small(caption, className="text-muted d-block mt-2") if caption else None,
                ]
            ),
        ],
        className="shadow-soft bg-dark-2 text-light h-100",
    )


def mpl_to_base64(fig: Figure, dpi: int = 170) -> str:
    """
    Serialize a Matplotlib Figure to base64 PNG. Assumes a valid Figure.
    """
    try:
        fig.patch.set_facecolor("#111111")
        for ax in fig.get_axes():
            ax.set_facecolor("#111111")
            ax.tick_params(colors="#E6E6E6")
            for spine in ax.spines.values():
                spine.set_color("#555555")
            if hasattr(ax, "title") and ax.title:
                ax.title.set_color("#E6E6E6")
            if ax.xaxis and ax.xaxis.label:
                ax.xaxis.label.set_color("#E6E6E6")
            if ax.yaxis and ax.yaxis.label:
                ax.yaxis.label.set_color("#E6E6E6")
    except Exception:
        pass

    buf = io.BytesIO()
    fig.canvas.draw()
    fig.savefig(
        buf, format="png", bbox_inches="tight", dpi=dpi,
        facecolor=fig.get_facecolor(), edgecolor=fig.get_facecolor()
    )
    plt.close(fig)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def _ensure_figure_from_result(res):
    """
    Accept common return shapes from your everything2 map functions:
      - Figure
      - (Figure, Axes...) tuple
      - Axes (use ax.figure)
      - None (fall back to current gcf if any)
    Raise if nothing usable exists.
    """
    if isinstance(res, Figure):
        return res
    if isinstance(res, tuple) and len(res) and isinstance(res[0], Figure):
        return res[0]
    try:
        import matplotlib.axes
        if isinstance(res, matplotlib.axes.Axes):
            return res.figure
    except Exception:
        pass
    if plt.get_fignums():
        return plt.gcf()
    raise RuntimeError("Plot function did not create or return a Matplotlib figure.")


def safe_make_mpl(title, fn, *args, **kwargs):
    try:
        res = fn(*args, **kwargs)
        fig = _ensure_figure_from_result(res)
        img_src = mpl_to_base64(fig)
        return img_card(title, img_src)
    except Exception as e:
        return error_card(title, f"Could not render map: {e}")


def safe_make_plotly(title, fn, *args, **kwargs):
    try:
        return fig_card(title, fn(*args, **kwargs))
    except Exception as e:
        return error_card(title, f"Could not render: {e}")


def error_card(title: str, msg: str):
    return dbc.Card(
        [
            dbc.CardHeader(title, className="card-title-dark"),
            dbc.CardBody(html.Div(f"⚠️ {msg}", style={"color": "#ff7b7b"})),
        ],
        className="shadow-soft bg-dark-2 text-light h-100",
    )


def two_up_grid(cards: List[dbc.Card]):
    cols = [dbc.Col(c, xs=12, md=6, className="mb-4") for c in cards]
    return dbc.Row(cols, className="g-3")


def section(title: str, subtitle: str = ""):
    return html.Div(
        [
            html.H2(title, className="section-title"),
            html.P(subtitle, className="section-subtitle") if subtitle else None,
            html.Hr(className="section-divider"),
        ]
    )


def subsection(title: str):
    return html.Div([html.H3(title, className="subsection-title")], className="mt-3")


# Load Data 
file_23 = "data/WomenElectorParticipation.xlsx"
file_24 = "data/WomenCandidateParticipation.xlsx"
file_33 = "data/ConstituencyWiseResult.xlsx"
file_6 = "data/StatewiseCandidateData.csv"
file_7 = "data/ConstituencyWiseSummary.csv"
file_7_2 = "data/ConstituencyWiseSummary2.csv"
file_8 = "data/CandidatesPerConstituency.csv"
file_assets = "data/LokSabhaAssetComparison.csv" 

df_women = india_elections_figures.load_women_participation_excel(file_23)
df_state_total, df_by_type = india_elections_figures.load_women_candidates_excel(file_24)
df_candidates, df_nota, df_results = india_elections_figures.load_constituency_detailed_results_excel(file_33)
df_results_nota = india_elections_figures.merge_nota_into_results(df_results, df_nota)

totals6, gender_long6, state_list6 = india_elections_figures.load_csv6_tidy(file_6)

df7, state_agg7, top_states7 = india_elections_figures.load_pc_level_csv7(file_7)
if "NonForfeiting" not in df7.columns:
    df7["NonForfeiting"] = (df7.get("Contestants", 0) - df7.get("FD", 0)).astype(float)
state_agg7_geo = india_elections_figures.attach_india_centroids(state_agg7)

df8, df8_long, df8_geo = india_elections_figures.load_candidate_distribution_csv8(file_8)

df_assets = india_elections_figures.load_asset_comparison_csv(file_assets)


# Build and cache MAP CARDS (interleaved later)
try:
    FEATURE_FILE = "indian-election-analysis-main/geo_utils/india_pc_2024_simplified.geojson"
    gdf = india_election_maps.load_geojson(feature_to_add_file=FEATURE_FILE)
    df_const = india_election_maps._prepare_voter_csv_for_deviation(file_7_2)

    # Build each map once, reuse across tabs/sections
    card_turnout_map           = safe_make_mpl("Voter Turnout — Map",           india_election_maps.plot_voter_turnout,             gdf=gdf)
    card_voter_dev_map         = safe_make_mpl("Voter Deviation — Map",         india_election_maps.plot_voter_deviation_map,       gdf, df_const)
    card_nota_map              = safe_make_mpl("NOTA % — Map",                  india_election_maps.plot_nota_map,                  gdf=gdf)
    card_alliance_map          = safe_make_mpl("Alliance — Map",                india_election_maps.plot_alliance_map,              gdf=gdf)
    card_margin_map            = safe_make_mpl("Victory Margin % — Map",        india_election_maps.plot_victory_margin_map,        gdf=gdf)
    card_women_seats_map       = safe_make_mpl("% Seats Won by Women — Map",    india_election_maps.plot_women_seats_map,           gdf=gdf, feature_to_add_file=FEATURE_FILE)
    card_female_electorate_map = safe_make_mpl("% Female Electorate — Map",     india_election_maps.plot_female_electorate_pct_map, gdf=gdf)

except Exception as e:
    diag = error_card("Maps", f"Failed to build maps: {e}")
    card_turnout_map = card_voter_dev_map = card_nota_map = card_alliance_map = card_margin_map = card_women_seats_map = card_female_electorate_map = diag


# Build Figure Groups (2-up with subsections)
# Overview
overview_cards_primary = [
    safe_make_plotly("1. Top 10 States/UTs by Total Electors", india_elections_figures.fig_1_top10_total_electors, df_women),
    safe_make_plotly("2. Top 10 States/UTs by Total Turnout %", india_elections_figures.fig_2_top10_total_turnout_pct, df_women),
]
overview_cards_comp = [
    safe_make_plotly("6. States Where Women’s Turnout % > Total Turnout %", india_elections_figures.fig_6_states_where_women_turnout_higher, df_women),
    safe_make_plotly("9. Turnout % vs Total Electors (Bubble = Seats)", india_elections_figures.fig_9_turnout_vs_total_electors_bubble, df_women),
]
overview_cards_share = [
    safe_make_plotly("7. % Electorate (Women) vs % Votes (Women)", india_elections_figures.fig_7_women_elector_pct_vs_votes_pct_scatter, df_women),
    safe_make_plotly("8. Share of Total Votes Polled (Top 10 vs Others)", india_elections_figures.fig_8_pie_share_total_votes_top10_vs_others, df_women),
]
overview_cards_maps = [
    card_turnout_map,
    card_voter_dev_map,
]

# Women
women_cards_top = [
    safe_make_plotly("11. Top 10 by Women Contestants", india_elections_figures.fig_11_top10_women_contestants, df_state_total),
    safe_make_plotly("12. Top 10 by Women Elected", india_elections_figures.fig_12_top10_women_elected, df_state_total),
]
women_cards_rates = [
    safe_make_plotly("15. Women Candidate Success Rate % by State", india_elections_figures.fig_15_state_success_rate_bar, df_state_total),
    safe_make_plotly("16. Women Candidate Deposit Forfeiture Rate % by State", india_elections_figures.fig_16_state_fd_rate_bar, df_state_total),
]
women_cards_rel = [
    safe_make_plotly("14. Total Women Elected by Constituency Type", india_elections_figures.fig_14_total_women_elected_by_const_type_bar, df_by_type),
    safe_make_plotly("17. Contestants vs Elected (Bubble, Trendline)", india_elections_figures.fig_17_contestants_vs_elected_scatter, df_state_total),
]
women_cards_vote = [
    safe_make_plotly("32. Vote Share by Gender (All Candidates) — Box", india_elections_figures.fig_32_vote_share_by_gender_box_all_candidates, df_candidates),
]
women_cards_maps = [
    card_women_seats_map,
    card_female_electorate_map,
]

# Constituency & Results
results_cards_ages = [
    safe_make_plotly("26. Winner Age by Party (Top 10) — Violin", india_elections_figures.fig_26_winner_age_by_party_violin_top10, df_results),
    safe_make_plotly("31. ALL Candidates Age — Violin (Top 10 Parties)", india_elections_figures.fig_31_candidate_age_violin_top10_parties, df_candidates),
]
# Removed Plot 34 from here; also remove this whole row to keep rows strictly 2-up
results_cards_maps_top = [
    card_alliance_map,
    card_margin_map,
]
# Removed single NOTA map row from Results; moved to Experimental to keep rows 2-up

# Nominations
nom_cards_heat_sankey = [
    safe_make_plotly("37. Rejection Rate Heatmap (Rejected/Filed)", india_elections_figures.fig_37_rejection_rate_heatmap_state_caste, totals6),
    safe_make_plotly("39. Nomination Flow Sankey — State Dropdown", india_elections_figures.fig_39_nomination_flow_sankey_dropdown, totals6, state_list6),
]
nom_cards_gap_spread = [
    safe_make_plotly("40. Contestants per Seat — Heatmap", india_elections_figures.fig_40_contestants_per_seat_heatmap, totals6),
    safe_make_plotly("48. Withdrawn Rate Distribution across States (by Caste)", india_elections_figures.fig_48_withdrawn_rate_strip_by_caste, totals6),
]
nom_cards_misc = [
    safe_make_plotly("42. State Leaderboard — RejectionRate (GEN)", india_elections_figures.fig_42_state_leaderboard_metric_by_caste, totals6),
    safe_make_plotly("46. Gender Gap in Contesting — (M − F) / TOTAL — GEN", india_elections_figures.fig_46_gender_gap_contesting_bar_by_caste, gender_long6),
]

# Turnout
turnout_cards_top = [
    safe_make_plotly("49. Turnout Leaderboard — Weighted by Electors (State)", india_elections_figures.fig_49_state_weighted_vtr_leaderboard_bar, state_agg7),
    safe_make_plotly("54. State Funnel — Nominations → Contestants → FD", india_elections_figures.fig_54_state_funnel_nominations_contestants_fd_dropdown, state_agg7),
]
turnout_cards_corr = [
    safe_make_plotly("59. Nominations vs Turnout (color = FD Rate)", india_elections_figures.fig_59_nominations_vs_vtr_pc_colored_fd_rate, df7),
    safe_make_plotly("60. Turnout Distributions: Top-6 vs Bottom-6 (PC-level)", india_elections_figures.fig_60_violin_turnout_top6_bottom6, df7, state_agg7),
]

# Distribution
dist_cards = [
    safe_make_plotly("69. Constituency Count Range Distribution (per State/UT)", india_elections_figures.fig_69_candidate_bucket_distribution_stackedbar, df8_long),
    safe_make_plotly("70. Ranking by Avg Candidates per Constituency", india_elections_figures.fig_70_avg_candidates_ranking_bar, df8),
]

# Assets
asset_cards = [
    safe_make_plotly("73. Top 10 Asset Increases & Decreases", india_elections_figures.fig_73_asset_increase_decrease_bar, df_assets)
]

# Cool / Experimental
cool_cards_1 = [
    safe_make_plotly("21. Winner Profile 5D Bubble", india_elections_figures.fig_21_winner_profile_5d_bubble, df_results),
    safe_make_plotly("22. Parallel Coordinates (Top 10 Parties)", india_elections_figures.fig_22_parallel_coords_top10_parties, df_results),
]
cool_cards_2 = [
    safe_make_plotly("23. 3D: Turnout % vs Age vs Margin %", india_elections_figures.fig_23_turnout_age_margin_3d_scatter, df_results),
    safe_make_plotly("25. SPLOM of Winner Metrics", india_elections_figures.fig_25_winner_metrics_splom, df_results),
]
cool_cards_3 = [
    safe_make_plotly("24. Treemap: Seats Won (Party→State→Category)", india_elections_figures.fig_24_treemap_seats_party_state_category, df_results),
    safe_make_plotly("29. Density Heatmap: Turnout vs Margin", india_elections_figures.fig_29_margin_vs_turnout_density_heatmap, df_results),
]
cool_cards_4 = [
    safe_make_plotly("43. Filed → Contesting (Dumbbell) — GEN", india_elections_figures.fig_43_dumbbell_filed_to_contesting_by_caste, totals6),
    safe_make_plotly("50. PC Turnout Distributions — Top-12 by Electors", india_elections_figures.fig_50_pc_vtr_distributions_top12_box, df7, top_states7),
]
cool_cards_5 = [
    safe_make_plotly("52. Contestants per PS vs VTR (PC-level)", india_elections_figures.fig_52_pc_contestants_per_ps_vs_vtr_scatter, df7),
    safe_make_plotly("53. Deposit Forfeiture Rate vs VTR (PC-level)", india_elections_figures.fig_53_pc_fd_rate_vs_vtr_scatter, df7),
]
cool_cards_6 = [
    safe_make_plotly("55. Treemap — Electors: State → PC (color = VTR)", india_elections_figures.fig_55_treemap_electors_state_pc_color_vtr, df7),
    safe_make_plotly("71. Avg vs Max Candidates — Bubble ~ Total", india_elections_figures.fig_71_avg_vs_max_candidates_scatter, df8),
]
# New pair to keep 2-per-row: moved Plot 34 + NOTA Map here
cool_cards_7 = [
    safe_make_plotly("34. Sunburst: All Candidates (Party → Category → Gender)", india_elections_figures.fig_34_sunburst_all_candidates_party_category_gender, df_candidates),
    card_nota_map,
]


# Dash App
external_stylesheets = [dbc.themes.CYBORG]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "India Elections — Dark Dashboard"


app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { background-color: #0e0e0f; }
            .shadow-soft { box-shadow: 0 8px 24px rgba(0,0,0,0.35); border: 1px solid #1e1e1f; }
            .bg-dark-2 { background-color: #151518 !important; }
            .card-title-dark { font-weight: 700; font-size: 1.05rem; letter-spacing: .2px; background-color: #0f0f12; }
            .section-title { color: #fff; font-weight: 800; letter-spacing: .5px; margin-top: 8px; }
            .section-subtitle { color: #c0c3c7; font-size: 0.95rem; }
            .subsection-title { color: #e2e5ea; font-weight: 700; margin-top: 10px; }
            .section-divider { border-top: 2px solid #202226; }
            .navbar-dark { background-color: #0b0b0c !important; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

header = dbc.Navbar(
    dbc.Container(
        [
            html.Div("🗳️", style={"fontSize": "26px", "marginRight": "8px"}),
            dbc.NavbarBrand("India Elections — Executive Dashboard", className="ms-1"),
            dbc.NavbarToggler(id="navbar-toggler"),
        ],
        fluid=True,
    ),
    color="dark",
    dark=True,
    sticky="top",
    className="navbar-dark"
)

tabs = dbc.Tabs(
    [
        dbc.Tab(
            label="Overview",
            tab_id="tab-overview",
            children=[
                html.Br(),
                section("Overview & Highlights", "Electors, turnout, and share patterns"),
                subsection("Core totals & rates"),
                two_up_grid(overview_cards_primary),
                subsection("Turnout maps"),
                two_up_grid(overview_cards_maps),
                subsection("Comparative slices"),
                two_up_grid(overview_cards_comp),
                subsection("Shares & composition"),
                two_up_grid(overview_cards_share),
            ],
        ),
        dbc.Tab(
            label="Women in Elections",
            tab_id="tab-women",
            children=[
                html.Br(),
                section("Women in Elections", "Candidates, outcomes, and rate diagnostics"),
                subsection("Top-line outcomes"),
                two_up_grid(women_cards_top),
                subsection("Rates & risks"),
                two_up_grid(women_cards_rates),
                subsection("Structure & relation"),
                two_up_grid(women_cards_rel),
                subsection("Geography of women’s representation & electorate"),
                two_up_grid(women_cards_maps),
                subsection("Vote share profile"),
                two_up_grid(women_cards_vote),
            ],
        ),
        dbc.Tab(
            label="Constituency & Results",
            tab_id="tab-results",
            children=[
                html.Br(),
                section("Constituency & Results", "Ages, performance, and composition"),
                subsection("Age distributions"),
                two_up_grid(results_cards_ages),
                subsection("Result geography"),
                two_up_grid(results_cards_maps_top),
                # Removed Performance & composition (single card after moving 34)
                # Removed Ballot features (NOTA map moved to Experimental)
            ],
        ),
        dbc.Tab(
            label="Nominations & Process",
            tab_id="tab-nom",
            children=[
                html.Br(),
                section("Nominations Pipeline", "From filings to withdrawals and gaps"),
                subsection("Heatmap & flow"),
                two_up_grid(nom_cards_heat_sankey),
                subsection("Gap & spread"),
                two_up_grid(nom_cards_gap_spread),
                subsection("Leaderboards & gaps"),
                two_up_grid(nom_cards_misc),
            ],
        ),
        dbc.Tab(
            label="Turnout & States",
            tab_id="tab-turnout",
            children=[
                html.Br(),
                section("Turnout & State Comparisons", "Rankings, funnels, and correlations"),
                subsection("Rank & funnel"),
                two_up_grid(turnout_cards_top),
                subsection("Correlations & spreads"),
                two_up_grid(turnout_cards_corr),
            ],
        ),
        dbc.Tab(
            label="Candidate Distribution",
            tab_id="tab-dist",
            children=[
                html.Br(),
                section("Candidate Distribution", "Counts & rankings"),
                two_up_grid(dist_cards),
            ],
        ),
        # --- NEW TAB ADDED HERE ---
        dbc.Tab(
            label="Assets of Candidates",
            tab_id="tab-assets",
            children=[
                html.Br(),
                section("Candidate Asset Changes", "Analysis of candidate wealth from asset comparison data"),
                subsection("Top Increases and Decreases"),
                two_up_grid(asset_cards),
            ],
        ),
        # --- END OF NEW TAB ---
        dbc.Tab(
            label="Experimental / Cool",
            tab_id="tab-cool",
            children=[
                html.Br(),
                section("Experimental / Cool Figures", "Visually rich exploratory views"),
                two_up_grid(cool_cards_1),
                two_up_grid(cool_cards_2),
                two_up_grid(cool_cards_3),
                two_up_grid(cool_cards_4),
                two_up_grid(cool_cards_5),
                two_up_grid(cool_cards_6),
                two_up_grid(cool_cards_7),  # includes moved Plot 34 + NOTA Map
            ],
        ),
    ],
    className="mt-3",
    active_tab="tab-overview",
)

footer = html.Div(
    [
        html.Hr(className="section-divider"),
        html.Div(
            "Indian Election Dashboard",
            className="text-muted",
            style={"fontSize": ".9rem"},
        ),
        html.Br(),
    ],
    className="mt-4",
)

app.layout = dbc.Container([header, tabs, footer], fluid=True, className="px-4")

if __name__ == "__main__":
    app.run(debug=False)
