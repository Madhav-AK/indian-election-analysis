import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re

COLUMN_MAP = {
    'STATE/UT': 'state',
    'NO OF SEATS': 'num_seats',
    'TOTAL ELECTORS': 'total_electors',
    'GENERAL ELECTORS': 'general_electors',
    'TOTAL WOMEN ELECTORS INCLUDING SERVICE ELECTORS': 'women_electors',
    '% OF WOMEN ELECTORS OVER TOTAL ELECTORS': 'women_electors_pct',
    'TOTAL VOTES POLLED': 'total_votes_polled',
    'TOTAL VOTES POLLED [Excluding Postal Votes]': 'total_votes_excl_postal',
    'TOTAL VOTES POLLED BY WOMEN [Excluding Postal Votes]': 'women_votes_polled',
    '% OF VOTES POLLED BY WOMEN [Excluding Postal Votes] OVER TOTAL VOTES POLLED (EXCLUDING Postal Votes)': 'women_votes_pct_of_total',
    '% OF WOMEN VOTERS OVER WOMEN ELECTORS': 'women_turnout_pct',
    'TOTAL POLL% IN THE STATE/UT': 'total_turnout_pct'
}

REQUIRED_COLS = [
    'state','num_seats','total_electors','women_electors','women_electors_pct',
    'total_votes_polled','women_votes_pct_of_total','women_turnout_pct','total_turnout_pct'
]

def load_women_participation_excel(file_path: str) -> pd.DataFrame:
    """
    Read the provided Excel (sheet structure from your notebook), rename columns, and basic clean.
    Returns a tidy DataFrame, one row per State/UT.
    """
    df = pd.read_excel(file_path, skiprows=1)
    df = df.rename(columns=COLUMN_MAP)

    # Drop rows where these key fields are missing (often totals/footers)
    df = df.dropna(subset=['state', 'num_seats', 'total_turnout_pct'])

    # Remove any total/sum rows that slipped through
    df = df[~df['state'].astype(str).str.contains('Total', case=False, na=False)].copy()

    # Ensure numeric types where appropriate (robustness against stray strings)
    numeric_cols = [c for c in REQUIRED_COLS if c not in ('state',)]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Final sanity drop of fully-empty key metrics
    df = df.dropna(subset=['total_electors', 'total_turnout_pct'])

    return df.reset_index(drop=True)

def with_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy with columns needed by some figures:
    - turnout_gap_women_vs_total = women_turnout_pct - total_turnout_pct
    - men_electors = total_electors - women_electors
    """
    pdf = df.copy()
    if 'women_turnout_pct' in pdf.columns and 'total_turnout_pct' in pdf.columns:
        pdf['turnout_gap_women_vs_total'] = pdf['women_turnout_pct'] - pdf['total_turnout_pct']
    if 'total_electors' in pdf.columns and 'women_electors' in pdf.columns:
        pdf['men_electors'] = pdf['total_electors'] - pdf['women_electors']
    return pdf


def fig_1_top10_total_electors(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Top-N states/UTs by total electors (horizontal bar, shades of red)."""
    pdf = df.sort_values('total_electors', ascending=True).tail(top_n)

    # custom red gradient: darker at low, lighter at high
    red_scale = [
        [0.0, "#330000"],
        [0.3, "#660000"],
        [0.6, "#990000"],
        [0.8, "#cc0000"],
        [1.0, "#ff3333"]
    ]

    fig = px.bar(
        pdf,
        x='total_electors',
        y='state',
        orientation='h',
        color='total_electors',
        color_continuous_scale=red_scale,
        title=f"Top {top_n} States/UTs by Total Electors",
        labels={'total_electors': 'Total Electors', 'state': 'State/UT'},
        template='plotly_dark'
    )
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=80, r=40, t=60, b=40),
        yaxis=dict(categoryorder='total ascending', title=None),
        xaxis_title="Total Electors",
        yaxis_title=None
    )
    return fig


def fig_2_top10_total_turnout_pct(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Top-N states/UTs by total turnout % (horizontal bar, shades of blue)."""
    pdf = df.sort_values('total_turnout_pct', ascending=True).tail(top_n)

    # custom blue gradient: darker at low, lighter at high
    blue_scale = [
        [0.0, "#001133"],
        [0.3, "#003366"],
        [0.6, "#005599"],
        [0.8, "#0077cc"],
        [1.0, "#33aaff"]
    ]

    fig = px.bar(
        pdf,
        x='total_turnout_pct',
        y='state',
        orientation='h',
        color='total_turnout_pct',
        color_continuous_scale=blue_scale,
        title=f"Top {top_n} States/UTs by Total Turnout (%)",
        labels={'total_turnout_pct': 'Total Turnout (%)', 'state': 'State/UT'},
        template='plotly_dark'
    )
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=80, r=40, t=60, b=40),
        yaxis=dict(categoryorder='total ascending', title=None),
        xaxis_title="Total Turnout (%)",
        yaxis_title=None
    )
    return fig


def fig_3_total_turnout_histogram(df: pd.DataFrame, nbins: int = 15) -> go.Figure:
    """Distribution of total turnout % (histogram)."""
    fig = px.histogram(
        df, x='total_turnout_pct', nbins=nbins,
        title='Distribution of Total Turnout % Across All States/UTs',
        labels={'total_turnout_pct': 'Total Turnout (%)'},
        template='plotly_dark'
    )
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    return fig

def fig_4_women_turnout_histogram(df: pd.DataFrame, nbins: int = 15) -> go.Figure:
    """Distribution of women turnout % (histogram)."""
    fig = px.histogram(
        df, x='women_turnout_pct', nbins=nbins,
        title='Distribution of Women Turnout % Across All States/UTs',
        labels={'women_turnout_pct': 'Women Turnout (%)'},
        template='plotly_dark'
    )
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    return fig

def fig_5_turnout_gap_histogram(df: pd.DataFrame, nbins: int = 15) -> go.Figure:
    """Distribution of (women turnout % - total turnout %) (histogram)."""
    pdf = with_derived_columns(df)
    fig = px.histogram(
        pdf, x='turnout_gap_women_vs_total', nbins=nbins,
        title='Distribution of Turnout Gap (Women % - Total %)',
        labels={'turnout_gap_women_vs_total': 'Gap (Women % - Total %)'},
        template='plotly_dark'
    )
    fig.update_layout(
        xaxis_title='Turnout Gap (Positive = Women > Total)',
        margin=dict(l=10, r=10, t=60, b=10)
    )
    return fig

def fig_6_states_where_women_turnout_higher(df: pd.DataFrame) -> go.Figure:
    """States where women's turnout % exceeded total turnout % (horizontal bar, clean labels)."""
    pdf = with_derived_columns(df)
    pdf = pdf[pdf['turnout_gap_women_vs_total'] > 0].sort_values(
        'turnout_gap_women_vs_total', ascending=True
    )

    fig = px.bar(
        pdf,
        x='turnout_gap_women_vs_total',
        y='state',
        orientation='h',
        color='turnout_gap_women_vs_total',
        title="States Where Women's Turnout % was HIGHER than Total Turnout %",
        labels={
            'turnout_gap_women_vs_total': 'Turnout Gap (Percentage Points)',
            'state': 'State/UT'
        },
        color_continuous_scale='Greens',
        template='plotly_dark'
    )

    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=80, r=40, t=60, b=40),
        yaxis=dict(categoryorder='total ascending', title=None),
        xaxis_title="Turnout Gap (Percentage Points)",
        yaxis_title=None
    )
    return fig

def fig_7_women_elector_pct_vs_votes_pct_scatter(df: pd.DataFrame, trendline: str = 'ols') -> go.Figure:
    """Scatter: % of electorate who are women vs % of votes polled by women (with trendline, larger dots)."""
    fig = px.scatter(
        df,
        x='women_electors_pct',
        y='women_votes_pct_of_total',
        hover_name='state',
        trendline=trendline,
        title='% Electorate (Women) vs. % Votes (Women)',
        labels={
            'women_electors_pct': '% of Electorate who are Women',
            'women_votes_pct_of_total': '% of Total Votes Polled by Women'
        },
        template='plotly_dark',
        size_max=16,
    )

    fig.update_traces(
        marker=dict(size=14, color='#66b3ff', line=dict(width=1, color='white')),
        selector=dict(mode='markers')
    )

    fig.update_layout(
        margin=dict(l=60, r=40, t=60, b=60),
        xaxis_title="% of Electorate who are Women",
        yaxis_title="% of Total Votes Polled by Women",
    )
    return fig

def fig_8_pie_share_total_votes_top10_vs_others(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Pie: share of total votes polled (Top-N states vs Others)."""
    pdf = df.sort_values('total_votes_polled', ascending=False).copy()
    top = pdf.head(top_n)[['state', 'total_votes_polled']].copy()
    others_sum = pdf.iloc[top_n:]['total_votes_polled'].sum()
    top.loc[len(top)] = {'state': 'All Others', 'total_votes_polled': others_sum}

    fig = px.pie(
        top, names='state', values='total_votes_polled',
        title=f'Share of Total Votes Polled (Top {top_n} States/UTs vs. Others)',
        template='plotly_dark'
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    return fig

def fig_9_turnout_vs_total_electors_bubble(df: pd.DataFrame) -> go.Figure:
    """Bubble: turnout % vs total electors (bubble size = num seats, log-x)."""
    fig = px.scatter(
        df, x='total_electors', y='total_turnout_pct', size='num_seats', color='state',
        hover_name='state', log_x=True,
        title='Turnout % vs. Total Electors (Bubble Size = Num Seats)',
        labels={
            'total_electors': 'Total Electors (Log Scale)',
            'total_turnout_pct': 'Total Turnout (%)',
            'num_seats': 'Number of Seats'
        },
        template='plotly_dark'
    )
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=60, b=10))
    return fig

def fig_10_stacked_men_vs_women_electors_top15_by_seats(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Stacked bars: men vs women electors for top-N states by number of seats."""
    pdf = with_derived_columns(df)
    pdf = pdf.sort_values('num_seats', ascending=False).head(top_n)
    melted = pdf.melt(
        id_vars='state', value_vars=['men_electors', 'women_electors'],
        var_name='Gender', value_name='Count'
    )
    fig = px.bar(
        melted, x='state', y='Count', color='Gender', barmode='stack',
        title=f'Men vs. Women Electors (Top {top_n} States by Seats)',
        template='plotly_dark',
        color_discrete_map={'men_electors': 'royalblue', 'women_electors': 'lightcoral'}
    )
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    return fig


# Women Candidates
def load_women_candidates_excel(file_path: str):
    """
    Reads '24-Participation-of-Women-Candidates.xlsx' (skiprows=2),
    cleans it, splits into:
      - df_state_total : rows where const_type == 'State Total' (state-level rollups)
      - df_by_type     : rows where const_type in {'GEN','SC','ST'}
    Adds:
      - calc_success_rate = 100 * women_elected / women_contestants
      - calc_fd_rate      = 100 * women_fd / women_contestants
    Returns (df_state_total, df_by_type).
    """
    df = pd.read_excel(file_path, skiprows=2)

    COLUMN_MAP = {
        'State /UT': 'state',
        'Seats': 'seats',
        'Constituency Type': 'const_type',
        'Contestants': 'women_contestants',
        'Elected': 'women_elected',
        'Deposits Forfeited': 'women_fd',
        'Over Total Women Candidates in the State': 'success_rate_given_pct',
        'Over total seats in State/UT': 'elected_vs_total_seats_pct'
    }
    df = df.rename(columns=COLUMN_MAP)

    # forward-fill state names (table-style sheet)
    df['state'] = df['state'].ffill()

    # drop non-data rows and 'Total' rollups
    df = df.dropna(subset=['const_type', 'women_contestants'])
    df = df[~df['state'].astype(str).str.contains('Total', case=False, na=False)].copy()

    # split
    df_state_total = df[df['const_type'] == 'State Total'].copy()
    df_by_type = df[df['const_type'] != 'State Total'].copy()

    # numeric safety
    for col in ['women_contestants', 'women_elected', 'women_fd', 'seats']:
        if col in df_state_total.columns:
            df_state_total[col] = pd.to_numeric(df_state_total[col], errors='coerce')
        if col in df_by_type.columns:
            df_by_type[col] = pd.to_numeric(df_by_type[col], errors='coerce')

    # derived rates (guard /0)
    df_state_total['calc_success_rate'] = (
        (df_state_total['women_elected'] / df_state_total['women_contestants']) * 100
    ).fillna(0)

    df_state_total['calc_fd_rate'] = (
        (df_state_total['women_fd'] / df_state_total['women_contestants']) * 100
    ).fillna(0)

    return df_state_total.reset_index(drop=True), df_by_type.reset_index(drop=True)


def fig_11_top10_women_contestants(df_state_total: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Top-N states/UTs by number of women contestants (horizontal bar, clean labels)."""
    pdf = df_state_total.sort_values('women_contestants', ascending=True).tail(top_n)
    fig = px.bar(
        pdf,
        x='women_contestants',
        y='state',
        orientation='h',
        color='women_contestants',
        title=f'Top {top_n} States/UTs by Number of Women Contestants',
        labels={'women_contestants': 'Women Contestants', 'state': 'State/UT'},
        template='plotly_dark',
        color_continuous_scale='Purples'
    )
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=80, r=40, t=60, b=40),
        yaxis=dict(categoryorder='total ascending', title=None),
        xaxis_title="Women Contestants",
        yaxis_title=None
    )
    return fig


def fig_12_top10_women_elected(df_state_total: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Top-N states/UTs by number of women elected (horizontal bar, clean labels)."""
    pdf = df_state_total.sort_values('women_elected', ascending=True).tail(top_n)
    fig = px.bar(
        pdf,
        x='women_elected',
        y='state',
        orientation='h',
        color='women_elected',
        title=f'Top {top_n} States/UTs by Number of Women Elected',
        labels={'women_elected': 'Women Elected', 'state': 'State/UT'},
        template='plotly_dark',
        color_continuous_scale='Magenta'
    )
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=80, r=40, t=60, b=40),
        yaxis=dict(categoryorder='total ascending', title=None),
        xaxis_title="Women Elected",
        yaxis_title=None
    )
    return fig

def fig_13_contestants_share_by_const_type_pie(df_by_type: pd.DataFrame) -> go.Figure:
    """Share of women contestants by constituency type (pie across all states)."""
    fig = px.pie(
        df_by_type, names='const_type', values='women_contestants',
        title='Share of Women Contestants by Constituency Type (All States)',
        labels={'const_type': 'Constituency Type', 'women_contestants': 'Contestants'},
        template='plotly_dark'
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    return fig

def fig_14_total_women_elected_by_const_type_bar(
    df_by_type: pd.DataFrame, 
    denom_col: str | None = None
) -> go.Figure:
    """
    Proportion of women elected by constituency (caste) type.
    Normalizes women_elected by a total per const_type.
    
    Parameters
    ----------
    df_by_type : DataFrame
        Must include at least: ['const_type', 'women_elected'] and one of the
        denominator columns listed below (or pass denom_col explicitly).
    denom_col : str | None
        Column to use as denominator. If None, the function will try (in order):
        ['total_elected','elected','total_seats','seats',
         'total_contesting','total_candidates','contestants'].
    """
    if denom_col is None:
        candidates = [
            'total_elected', 'elected', 'total_seats', 'seats',
            'total_contesting', 'total_candidates', 'contestants'
        ]
        denom_col = next((c for c in candidates if c in df_by_type.columns), None)
        if denom_col is None:
            raise ValueError(
                "Could not infer a denominator column. "
                "Pass denom_col explicitly (e.g., 'total_elected')."
            )

    # Aggregate by constituency type
    grouped = (
        df_by_type
        .groupby('const_type', as_index=False)
        .agg({'women_elected': 'sum', denom_col: 'sum'})
        .rename(columns={denom_col: 'denom_total'})
    )

    # Compute normalized ratio; handle divide-by-zero safely
    grouped['women_ratio'] = grouped.apply(
        lambda r: (r['women_elected'] / r['denom_total']) if r['denom_total'] not in (0, None) else float('nan'),
        axis=1
    )

    # Build bar chart
    fig = px.bar(
        grouped,
        x='const_type',
        y='women_ratio',
        title='Proportion of Women Elected by Constituency (Caste) Type',
        labels={'women_ratio': 'Women Elected (%)', 'const_type': 'Constituency Type'},
        template='plotly_dark',
        text=grouped['women_ratio'].map(lambda v: f"{v:.1%}" if pd.notna(v) else "")
    )

    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    return fig

def fig_15_state_success_rate_bar(df_state_total: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Top-N states/UTs by women candidate success rate % (Elected / Contestants)."""
    pdf = df_state_total.sort_values('calc_success_rate', ascending=False).head(top_n)

    green_scale = [
        [0.0, "#003300"],
        [0.3, "#006600"],
        [0.6, "#009933"],
        [0.8, "#33cc33"],
        [1.0, "#99ff99"]
    ]

    fig = px.bar(
        pdf,
        x='calc_success_rate',
        y='state',
        orientation='h',
        color='calc_success_rate',
        color_continuous_scale=green_scale,
        title=f'Top {top_n} States/UTs by Women Candidate Success Rate %',
        labels={'calc_success_rate': 'Success Rate (%)', 'state': 'State/UT'},
        template='plotly_dark'
    )
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=80, r=40, t=60, b=40),
        yaxis=dict(categoryorder='total ascending', title=None),
        xaxis=dict(title="Success Rate (%)", ticksuffix="%")
    )
    return fig


def fig_16_state_fd_rate_bar(df_state_total: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Top-N states/UTs by women candidate deposit forfeiture (FD) rate %."""
    pdf = df_state_total.sort_values('calc_fd_rate', ascending=False).head(top_n)

    red_scale = [
        [0.0, "#330000"],
        [0.3, "#660000"],
        [0.6, "#990000"],
        [0.8, "#cc0000"],
        [1.0, "#ff6666"]
    ]

    fig = px.bar(
        pdf,
        x='calc_fd_rate',
        y='state',
        orientation='h',
        color='calc_fd_rate',
        color_continuous_scale=red_scale,
        title=f'Top {top_n} States/UTs by Women Candidate FD Rate %',
        labels={'calc_fd_rate': 'FD Rate (%)', 'state': 'State/UT'},
        template='plotly_dark'
    )
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=80, r=40, t=60, b=40),
        yaxis=dict(categoryorder='total ascending', title=None),
        xaxis=dict(title="FD Rate (%)", ticksuffix="%")
    )
    return fig


def fig_17_contestants_vs_elected_scatter(df_state_total: pd.DataFrame, trendline: str = 'ols') -> go.Figure:
    """Scatter: women contestants vs women elected (bubble size=seats, with trendline)."""
    fig = px.scatter(
        df_state_total, x='women_contestants', y='women_elected', size='seats',
        hover_name='state', trendline=trendline,
        title='Women Contestants vs Women Elected (by State/UT)',
        labels={'women_contestants': 'Women Contestants', 'women_elected': 'Women Elected'},
        template='plotly_dark'
    )
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    return fig


def fig_18_treemap_elected_by_state_and_type(df_by_type: pd.DataFrame) -> go.Figure:
    """Treemap of women elected by state and constituency type (filtering zeros)."""
    pdf = df_by_type[df_by_type['women_elected'] > 0].copy()
    fig = px.treemap(
        pdf, path=[px.Constant("All Elected Women"), 'state', 'const_type'],
        values='women_elected', color='const_type',
        title='Treemap of Women Elected by State and Constituency Type',
        labels={'women_elected': 'Women Elected'},
        template='plotly_dark'
    )
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    return fig


def fig_19_success_rate_histogram(df_state_total: pd.DataFrame, nbins: int = 15) -> go.Figure:
    """Histogram: distribution of women candidate success rates across states."""
    fig = px.histogram(
        df_state_total, x='calc_success_rate', nbins=nbins,
        title='Distribution of Women Candidate Success Rates (Across States)',
        labels={'calc_success_rate': 'Success Rate (%)'},
        template='plotly_dark'
    )
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    return fig


def fig_20_fd_rate_histogram(df_state_total: pd.DataFrame, nbins: int = 15) -> go.Figure:
    """Histogram: distribution of women candidate deposit forfeiture (FD) rates across states."""
    fig = px.histogram(
        df_state_total, x='calc_fd_rate', nbins=nbins,
        title='Distribution of Women Candidate Deposit Forfeited (FD) Rates',
        labels={'calc_fd_rate': 'FD Rate (%)'},
        template='plotly_dark'
    )
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    return fig


# Constituency-wise Detailed Results

def load_constituency_detailed_results_excel(file_path: str):
    """
    Reads '33-Constituency-Wise-Detailed-Result.xlsx' (skiprows=2),
    cleans columns, splits into:
      - df_candidates : all candidate rows except NOTA (numeric coerced, essentials kept)
      - df_nota       : rows where party_name == 'NOTA'
      - df_results    : one row per (state, pc_name) for the winner, merged with runner-up to compute:
            * margin_pct           = pct_over_polled_win - pct_over_polled_run
            * voter_turnout_pct    = 100 * total_votes_in_pc / total_electors
            * seat_count           = 1 (for treemaps)
    Returns (df_candidates, df_nota, df_results).
    """
    raw = pd.read_excel(file_path, skiprows=2)

    COLUMN_MAP = {
        'State Name': 'state',
        'PC Name': 'pc_name',
        'Candidate Name': 'candidate_name',
        'Gender': 'gender',
        'Age': 'age',
        'Category': 'category',
        'Party Name': 'party_name',
        'Party Symbol': 'party_symbol',
        'Total Votes Polled In\nThe Constituency': 'total_votes_in_pc',
        'Valid Votes': 'valid_votes_in_pc',
        'General': 'general_votes',
        'Postal': 'postal_votes',
        'Total': 'total_votes',
        '% of Votes Secured': 'pct_of_votes_secured',
        'Over Total Electors In Constituency': 'pct_over_electors',
        'Over Total Votes Polled In Constituency': 'pct_over_polled',
        'Over Total Valid Votes Polled In Constituency': 'pct_over_valid',
        'Total Electors': 'total_electors'
    }
    df = raw.rename(columns=COLUMN_MAP)

    # Separate NOTA vs candidates
    df_nota = df[df['party_name'] == 'NOTA'].copy()
    df_candidates = df[df['party_name'] != 'NOTA'].copy()

    # Numeric coercion on candidate columns
    numeric_cols = [
        'age','total_votes_in_pc','valid_votes_in_pc','general_votes','postal_votes',
        'total_votes','pct_over_electors','pct_over_polled','pct_over_valid','total_electors'
    ]
    for c in numeric_cols:
        if c in df_candidates.columns:
            df_candidates[c] = pd.to_numeric(df_candidates[c], errors='coerce')

    # Basic cleanliness
    df_candidates = df_candidates.dropna(subset=['total_votes', 'pc_name', 'state']).copy()

    # Rank within each constituency to find winners/runners-up
    df_candidates = df_candidates.sort_values(
        by=['state','pc_name','total_votes'], ascending=[True, True, False]
    )

    # winners: first row per (state, pc_name)
    df_winners = df_candidates.groupby(['state','pc_name'], as_index=False).first()

    # runners-up: second row per (state, pc_name)
    df_runners_up = (
        df_candidates.groupby(['state','pc_name']).nth(1).reset_index()[['state','pc_name','total_votes','pct_over_polled']]
    )

    # Merge winners + runner-up metrics
    df_results = pd.merge(
        df_winners,
        df_runners_up,
        on=['state','pc_name'],
        suffixes=('_win','_run')  # e.g., total_votes_win, total_votes_run, pct_over_polled_win, pct_over_polled_run
    )

    # Derived fields
    df_results['margin_pct'] = df_results['pct_over_polled_win'] - df_results['pct_over_polled_run']
    df_results['voter_turnout_pct'] = (df_results['total_votes_in_pc'] / df_results['total_electors']) * 100
    df_results['seat_count'] = 1

    # Ensure NOTA percent numeric for its histogram later
    if 'pct_over_polled' in df_nota.columns:
        df_nota['pct_over_polled'] = pd.to_numeric(df_nota['pct_over_polled'], errors='coerce')

    return df_candidates.reset_index(drop=True), df_nota.reset_index(drop=True), df_results.reset_index(drop=True)


def fig_21_winner_profile_5d_bubble(df_results: pd.DataFrame) -> go.Figure:
    """
    5D bubble: x=winner age, y=winner vote share (% over polled),
    size=margin %, color=party, hover=candidate/state/pc.
    """
    fig = px.scatter(
        df_results,
        x='age', y='pct_over_polled_win', size='margin_pct', color='party_name',
        hover_name='candidate_name',
        hover_data=['state','pc_name','party_name','margin_pct'],
        title='Winner Profile: Age vs Vote Share (%), Bubble Size=Margin %, Color=Party',
        labels={'age':'Winner Age','pct_over_polled_win':'Winner Vote Share (%)','margin_pct':'Margin (%)','party_name':'Winner Party'},
        template='plotly_dark'
    )
    fig.update_layout(margin=dict(l=10, r=10, t=70, b=10))
    return fig


def fig_22_parallel_coords_top10_parties(df_results: pd.DataFrame) -> go.Figure:
    """
    Parallel coordinates of winner metrics for top-10 parties by seats.
    Dimensions: age, vote share %, margin %, turnout %, total electors.
    """
    top_10 = df_results['party_name'].value_counts().head(10).index.tolist()
    pdf = df_results[df_results['party_name'].isin(top_10)].copy()
    fig = px.parallel_coordinates(
        pdf,
        dimensions=['age','pct_over_polled_win','margin_pct','voter_turnout_pct','total_electors'],
        color='pct_over_polled_win',
        color_continuous_scale=px.colors.sequential.Viridis,
        labels={'age':'Age','pct_over_polled_win':'Vote Share %','margin_pct':'Margin %','voter_turnout_pct':'Turnout %','total_electors':'Total Electors'},
        title='Parallel Coordinates: Winners (Top 10 Parties)',
        template='plotly_dark'
    )
    fig.update_layout(margin=dict(l=10, r=10, t=70, b=10))
    return fig


def fig_23_turnout_age_margin_3d_scatter(df_results: pd.DataFrame) -> go.Figure:
    """3D scatter: (turnout %, age, margin %) colored by winner category."""
    fig = px.scatter_3d(
        df_results,
        x='voter_turnout_pct', y='age', z='margin_pct',
        color='category',
        hover_name='pc_name',
        title='3D: Turnout % vs Winner Age vs Margin %',
        labels={'voter_turnout_pct':'Voter Turnout %','age':'Winner Age','margin_pct':'Margin %','category':'Category'},
        template='plotly_dark'
    )
    fig.update_layout(margin=dict(l=10, r=10, t=70, b=10))
    return fig


def fig_24_treemap_seats_party_state_category(df_results: pd.DataFrame) -> go.Figure:
    """Treemap of seats won: Party → State → Category."""
    fig = px.treemap(
        df_results,
        path=[px.Constant("All Seats"), 'party_name', 'state', 'category'],
        values='seat_count',
        color='category',
        title='Treemap: Seats Won (Party → State → Category)',
        template='plotly_dark'
    )
    fig.update_layout(margin=dict(l=10, r=10, t=70, b=10))
    return fig


def fig_25_winner_metrics_splom(df_results: pd.DataFrame) -> go.Figure:
    """Scatter-plot matrix (SPLOM) of winner metrics, colored by gender."""
    fig = px.scatter_matrix(
        df_results,
        dimensions=['age','pct_over_polled_win','margin_pct','voter_turnout_pct'],
        color='gender',
        title='SPLOM: Winner Metrics',
        labels={'age':'Age','pct_over_polled_win':'Vote %','margin_pct':'Margin %','voter_turnout_pct':'Turnout %','gender':'Gender'},
        template='plotly_dark'
    )
    fig.update_layout(margin=dict(l=10, r=10, t=70, b=10))
    return fig


def fig_26_winner_age_by_party_violin_top10(df_results: pd.DataFrame) -> go.Figure:
    """Violin of winner age by party (top-10 parties by seats), with box + points."""
    top_10 = df_results['party_name'].value_counts().head(10).index.tolist()
    pdf = df_results[df_results['party_name'].isin(top_10)].copy()
    fig = px.violin(
        pdf,
        y='age', x='party_name', color='party_name',
        box=True, points='all',
        title='Winner Age by Party (Top 10 Parties)',
        labels={'age':'Winner Age','party_name':'Winner Party'},
        template='plotly_dark'
    )
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=70, b=10))
    return fig


def fig_27_nota_vote_share_histogram(df_nota: pd.DataFrame, nbins: int = 50) -> go.Figure:
    """Histogram: distribution of NOTA vote share (% over total votes polled)."""
    fig = px.histogram(
        df_nota,
        x='pct_over_polled', nbins=nbins,
        title='Distribution of NOTA Vote Share (%)',
        labels={'pct_over_polled':'NOTA Vote Share (%)'},
        template='plotly_dark'
    )
    fig.update_layout(margin=dict(l=10, r=10, t=70, b=10))
    return fig


def fig_28_independent_vote_share_histogram(df_candidates: pd.DataFrame, nbins: int = 50) -> go.Figure:
    """Histogram: distribution of Independent candidates' vote share (%), log-scaled Y for long tail."""
    pdf = df_candidates[df_candidates['party_name'] == 'IND'].copy()
    fig = px.histogram(
        pdf,
        x='pct_over_polled', nbins=nbins,
        title='Independent Candidates: Vote Share (%) (Log Scale)',
        labels={'pct_over_polled':'Independent Vote Share (%)'},
        template='plotly_dark',
    )
    fig.update_yaxes(type='log')
    fig.update_layout(margin=dict(l=10, r=10, t=70, b=10))
    return fig

# Constituency-wise

def merge_nota_into_results(df_results: pd.DataFrame, df_nota: pd.DataFrame) -> pd.DataFrame:
    """
    Adds NOTA percentage (as 'nota_pct') to df_results via (state, pc_name) merge.
    If df_nota already has 'nota_pct', it uses that; otherwise expects 'pct_over_polled'.
    """
    nota = df_nota.copy()
    if 'nota_pct' not in nota.columns and 'pct_over_polled' in nota.columns:
        nota = nota.rename(columns={'pct_over_polled': 'nota_pct'})
    keep = ['state', 'pc_name', 'nota_pct']
    nota = nota[[c for c in keep if c in nota.columns]].copy()
    return df_results.merge(nota, on=['state', 'pc_name'], how='left')


def top_n_parties(series_or_df, n: int = 10, party_col: str = 'party_name') -> list:
    """
    Returns top-N party names by frequency (works with either a Series of parties or a DataFrame with party_name column).
    """
    if isinstance(series_or_df, pd.Series):
        return series_or_df.value_counts().head(n).index.tolist()
    return series_or_df[party_col].value_counts().head(n).index.tolist()

def fig_29_margin_vs_turnout_density_heatmap(df_results: pd.DataFrame, nbinsx: int = 30, nbinsy: int = 30) -> go.Figure:
    """
    Density heatmap: voter_turnout_pct vs margin_pct.
    """
    fig = px.density_heatmap(
        df_results,
        x='voter_turnout_pct', y='margin_pct',
        nbinsx=nbinsx, nbinsy=nbinsy,
        title='Density Heatmap: Voter Turnout (%) vs Margin of Victory (%)',
        labels={'voter_turnout_pct': 'Voter Turnout (%)', 'margin_pct': 'Margin of Victory (%)'},
        template='plotly_dark'
    )
    fig.update_layout(margin=dict(l=10, r=10, t=70, b=10))
    return fig


def fig_30_winner_vs_runner_vote_share_scatter(df_results: pd.DataFrame) -> go.Figure:
    """
    Scatter: winner vote share (%) vs runner-up vote share (%),
    bubble size = margin %, color = party.
    Adds diagonal y=x reference.
    Only BJP and INC are visible by default.
    """
    fig = px.scatter(
        df_results,
        x='pct_over_polled_win',
        y='pct_over_polled_run',
        size='margin_pct',
        color='party_name',
        hover_name='pc_name',
        hover_data=['state', 'candidate_name', 'margin_pct'],
        title='Winner Vote Share (%) vs Runner-Up Vote Share (%)',
        labels={
            'pct_over_polled_win': 'Winner Vote Share (%)',
            'pct_over_polled_run': 'Runner-Up Vote Share (%)'
        },
        template='plotly_dark'
    )

    # Diagonal y=x up to the observed max of both axes
    vmax = pd.concat([df_results['pct_over_polled_win'], df_results['pct_over_polled_run']]).max(skipna=True)
    vmax = 0 if pd.isna(vmax) else float(vmax)
    fig.add_shape(
        type="line", x0=0, y0=0, x1=vmax, y1=vmax,
        line=dict(color="gray", width=2, dash="dash")
    )

    # Show only BJP and INC traces by default
    visible_parties = {"BJP", "INC"}
    for trace in fig.data:
        if hasattr(trace, "name") and trace.name not in visible_parties:
            trace.visible = "legendonly"

    fig.update_layout(
        margin=dict(l=10, r=10, t=70, b=10),
        legend_title_text="Party",
    )
    return fig


def fig_31_candidate_age_violin_top10_parties(df_candidates: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Violin of ALL candidates' ages by party (limited to top-N parties by candidate count).
    """
    parties = top_n_parties(df_candidates, n=top_n, party_col='party_name')
    pdf = df_candidates[df_candidates['party_name'].isin(parties)].copy()
    fig = px.violin(
        pdf,
        y='age', x='party_name', color='party_name',
        box=True,
        title=f'Age Distribution of ALL Candidates (Top {top_n} Parties)',
        labels={'age': 'Candidate Age', 'party_name': 'Party'},
        template='plotly_dark'
    )
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=70, b=10))
    return fig


def fig_32_vote_share_by_gender_box_all_candidates(df_candidates: pd.DataFrame, log_y: bool = True) -> go.Figure:
    """
    Box plot of candidate vote share (%) by gender (all candidates).
    Uses log scale on Y to show long tail if log_y=True.
    """
    fig = px.box(
        df_candidates,
        x='gender', y='pct_over_polled', color='gender',
        points='outliers', notched=False,
        title='Vote Share (%) Distribution by Gender [Note: Y axis is log scale]',
        labels={'gender': 'Gender', 'pct_over_polled': 'Vote Share Secured (%)'},
        template='plotly_dark'
    )
    if log_y:
        fig.update_yaxes(type="log")
    fig.update_layout(margin=dict(l=10, r=10, t=70, b=10))
    return fig


def fig_33_nota_vs_margin_scatter(df_results_with_nota: pd.DataFrame, trendline: str = 'ols') -> go.Figure:
    """
    Scatter: NOTA % vs margin of victory %, colored by voter turnout % (continuous),
    with optional trendline (default 'ols').
    NOTE: Pass df_results already merged with NOTA via merge_nota_into_results(...).
    """
    fig = px.scatter(
        df_results_with_nota,
        x='margin_pct', y='nota_pct',
        color='voter_turnout_pct', color_continuous_scale='Viridis',
        trendline=trendline,
        hover_name='pc_name',
        title='NOTA % vs Margin of Victory %',
        labels={'margin_pct': 'Margin of Victory (%)', 'nota_pct': 'NOTA Vote Share (%)', 'voter_turnout_pct': 'Voter Turnout (%)'},
        template='plotly_dark'
    )
    fig.update_layout(margin=dict(l=10, r=10, t=70, b=10))
    return fig


def fig_34_sunburst_all_candidates_party_category_gender(df_candidates: pd.DataFrame, maxdepth: int = 2) -> go.Figure:
    """
    Sunburst: All candidates by Party → Category → Gender.
    """
    pdf = df_candidates.copy()
    pdf['__count'] = 1
    fig = px.sunburst(
        pdf,
        path=[px.Constant("All Candidates"), 'party_name', 'category', 'gender'],
        values='__count',
        title='All Candidates by Party, Category, and Gender',
        template='plotly_dark',
        maxdepth=maxdepth
    )
    fig.update_traces(textinfo="label+percent parent")
    fig.update_layout(margin=dict(l=10, r=10, t=70, b=10))
    return fig


def load_csv6_tidy(csv_path: str = "6.csv"):
    """
    Reads the CSV with repeating [M, F, TG, TOTAL] blocks for metrics:
      FILED, REJECTED, WITHDRAWN, CONTESTING, FORFEITED
    Returns:
      totals: per (State, ConstituencyType) wide table with derived columns
      gender_long: long table ([State, ConstituencyType, Metric, Gender, Value])
      state_list: list of states including 'ALL-INDIA'
    """
    raw = pd.read_csv(csv_path)

    metrics = ["FILED", "REJECTED", "WITHDRAWN", "CONTESTING", "FORFEITED"]
    genders = ["M", "F", "TG", "TOTAL"]

    # Rename repeating columns to METRIC_GENDER ordered by appearance
    tidy_df = raw.copy()
    idx = 3  # after first three fixed columns
    for met in metrics:
        for gen in genders:
            if idx < len(tidy_df.columns):
                tidy_df.rename(columns={tidy_df.columns[idx]: f"{met}_{gen}"}, inplace=True)
                idx += 1

    # Standardize base columns
    tidy_df.rename(columns={
        "STATE": "State",
        "CONSTITUENCY TYPE": "ConstituencyType",
        "NO. OF SEATS": "Seats"
    }, inplace=True)

    # Long form
    value_vars = [f"{m}_{g}" for m in metrics for g in genders if f"{m}_{g}" in tidy_df.columns]
    long = tidy_df.melt(
        id_vars=["State", "ConstituencyType", "Seats"],
        value_vars=value_vars,
        var_name="MetricGender",
        value_name="Value"
    )
    long[["Metric", "Gender"]] = long["MetricGender"].str.split("_", expand=True)
    long.drop(columns=["MetricGender"], inplace=True)

    # Numeric safety
    long["Value"] = pd.to_numeric(long["Value"], errors="coerce").fillna(0)
    tidy_df["Seats"] = pd.to_numeric(tidy_df["Seats"], errors="coerce").fillna(0)

    # Per-state-caste totals (TOTAL gender)
    totals = long[long["Gender"] == "TOTAL"].pivot_table(
        index=["State", "ConstituencyType"],
        columns="Metric",
        values="Value",
        aggfunc="sum",
        fill_value=0
    ).reset_index()

    # Derived columns
    totals["ACCEPTED"] = totals["FILED"] - totals["REJECTED"]
    totals["WithdrawnRate"] = np.where(totals["ACCEPTED"] > 0, totals["WITHDRAWN"] / totals["ACCEPTED"], 0.0)
    totals["RejectionRate"] = np.where(totals["FILED"] > 0, totals["REJECTED"] / totals["FILED"], 0.0)
    totals["ContestRatio"] = np.where(totals["FILED"] > 0, totals["CONTESTING"] / totals["FILED"], 0.0)

    # Bring Seats back and compute contestants/seat
    totals = totals.merge(
        tidy_df[["State", "ConstituencyType", "Seats"]].drop_duplicates(),
        on=["State", "ConstituencyType"],
        how="left"
    )
    totals["ContestantsPerSeat"] = np.where(totals["Seats"] > 0, totals["CONTESTING"] / totals["Seats"], np.nan)

    # Clean constituency type order
    if "ConstituencyType" in totals.columns:
        totals["ConstituencyType"] = pd.Categorical(totals["ConstituencyType"], ["GEN", "SC", "ST"], ordered=True)

    gender_long = long.copy()
    state_list = ["ALL-INDIA"] + sorted(totals["State"].unique().tolist())

    return totals, gender_long, state_list


def _csv6_filter_state_totals(totals: pd.DataFrame, state: str) -> pd.DataFrame:
    """Aggregates over states when state == 'ALL-INDIA', otherwise filters."""
    if state == "ALL-INDIA":
        cols_to_sum = [c for c in totals.columns if c not in ["State", "ConstituencyType", "Seats"]]
        agg = totals.groupby(["ConstituencyType"], as_index=False)[cols_to_sum].sum(numeric_only=True)
        agg["State"] = "ALL-INDIA"
        return agg[["State", "ConstituencyType"] + cols_to_sum]
    return totals[totals["State"] == state].copy()


def _csv6_gender_agg(gender_long: pd.DataFrame, state: str, metric: str = "FORFEITED"):
    """Aggregates gender_long to (ConstituencyType, Gender) sums for a given state (or ALL-INDIA)."""
    g = gender_long.copy()
    if state != "ALL-INDIA":
        g = g[g["State"] == state]
    g = g[(g["Metric"] == metric) & (g["Gender"].isin(["M", "F", "TG"]))]
    g = g.groupby(["ConstituencyType", "Gender"], as_index=False)["Value"].sum()
    g["ConstituencyType"] = pd.Categorical(g["ConstituencyType"], ["GEN", "SC", "ST"], ordered=True)
    g = g.sort_values(["ConstituencyType", "Gender"])
    return g


def fig_35_rejected_vs_accepted_by_caste_dropdown(totals: pd.DataFrame, state_list: list[str]) -> go.Figure:
    """Grouped bars: Accepted vs Rejected across constituency type; dropdown to switch state / ALL-INDIA."""
    frames = {st: _csv6_filter_state_totals(totals, st).sort_values(["ConstituencyType"]) for st in state_list}
    init_state = state_list[0]
    base = frames[init_state]
    x = base["ConstituencyType"]

    fig = go.Figure()
    fig.add_bar(name="Accepted (TOTAL)", x=x, y=base["ACCEPTED"])
    fig.add_bar(name="Rejected (TOTAL)", x=x, y=base["REJECTED"])

    buttons = []
    for st in state_list:
        df = frames[st]
        buttons.append(dict(
            label=st,
            method="update",
            args=[{"y": [df["ACCEPTED"], df["REJECTED"]],
                   "x": [df["ConstituencyType"], df["ConstituencyType"]]},
                  {"title": f"Rejected vs Accepted — {st}"}]
        ))

    fig.update_layout(
        barmode="group",
        title=f"Rejected vs Accepted — {init_state}",
        xaxis_title="Constituency Type (GEN / SC / ST)",
        yaxis_title="Count (TOTAL)",
        legend_title="Metric",
        updatemenus=[dict(type="dropdown", x=1.15, y=1.0, xanchor="left", buttons=buttons, showactive=True)],
        margin=dict(r=160)
    )
    return fig


def fig_36_forfeited_by_gender_caste_dropdown(gender_long: pd.DataFrame, state_list: list[str]) -> go.Figure:
    """Stacked bars: FORFEITED by gender within each constituency type; dropdown to switch state / ALL-INDIA."""
    frames = {st: _csv6_gender_agg(gender_long, st, metric="FORFEITED") for st in state_list}
    init_state = state_list[0]
    base = frames[init_state]
    x = base["ConstituencyType"].cat.categories.tolist() if hasattr(base["ConstituencyType"], 'cat') else base["ConstituencyType"].unique().tolist()

    fig = go.Figure()
    for gen in ["M", "F", "TG"]:
        vals = base[base["Gender"] == gen].set_index("ConstituencyType").reindex(x)["Value"]
        fig.add_bar(name=gen, x=x, y=vals)

    buttons = []
    for st in state_list:
        g = frames[st]
        vals_M = g[g["Gender"] == "M"].set_index("ConstituencyType").reindex(x)["Value"]
        vals_F = g[g["Gender"] == "F"].set_index("ConstituencyType").reindex(x)["Value"]
        vals_TG = g[g["Gender"] == "TG"].set_index("ConstituencyType").reindex(x)["Value"]
        buttons.append(dict(
            label=st,
            method="update",
            args=[{"y": [vals_M, vals_F, vals_TG]},
                  {"title": f"Deposit Forfeited (Gender breakdown) — {st}"}]
        ))

    fig.update_layout(
        barmode="stack",
        title=f"Deposit Forfeited (Gender breakdown) — {init_state}",
        xaxis_title="Constituency Type",
        yaxis_title="Forfeited (count)",
        legend_title="Gender",
        updatemenus=[dict(type="dropdown", x=1.15, y=1.0, xanchor="left", buttons=buttons, showactive=True)],
        margin=dict(r=160)
    )
    return fig


def fig_37_rejection_rate_heatmap_state_caste(totals: pd.DataFrame) -> go.Figure:
    """Clean, compact Heatmap: RejectionRate (Rejected / Filed) by State × ConstituencyType."""
    # Pivot data
    heat = totals.pivot_table(
        index="State",
        columns="ConstituencyType",
        values="RejectionRate",
        aggfunc="mean"
    )

    # Ensure all columns exist
    for col in ["GEN", "SC", "ST"]:
        if col not in heat.columns:
            heat[col] = float("nan")

    # Sort by mean rate
    heat = heat[["GEN", "SC", "ST"]]
    heat["mean_rate"] = heat.mean(axis=1, skipna=True)
    heat = heat.sort_values("mean_rate", ascending=False).drop(columns=["mean_rate"])

    # Compact dimensions
    num_states = len(heat.index)
    calculated_height = max(350, num_states * 12 + 120)
    calculated_width = 750

    # Plot heatmap
    fig = px.imshow(
        heat,
        labels=dict(x="Constituency Type", y="State", color="Rejection Rate"),
        x=["GEN", "SC", "ST"],
        y=heat.index,
        title="Rejection Rate Heatmap (Sorted by Average Rate)",
        color_continuous_scale="Reds",
        aspect="auto",
    )

    # Style cleanup
    fig.update_coloraxes(colorbar_title="Rate")
    fig.update_layout(
        height=calculated_height,
        width=calculated_width,
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=60),
    )

    # ✅ Remove background grid and axis lines
    fig.update_xaxes(showgrid=False, zeroline=False, tickangle=0, tickfont=dict(size=12))
    fig.update_yaxes(showgrid=False, zeroline=False, categoryorder="array", categoryarray=heat.index)

    return fig



def fig_38_nom_to_contest_ratio_vs_contestants_per_seat_scatter(totals: pd.DataFrame) -> go.Figure:
    """
    Scatter: (Filed / Contesting) vs Contestants per Seat, colored by ConstituencyType,
    with rich hover on nomination funnel counts.
    """
    scatter_df = totals.copy()
    scatter_df["Nom_to_Contest_Ratio"] = np.where(
        scatter_df["CONTESTING"] > 0, scatter_df["FILED"] / scatter_df["CONTESTING"], np.nan
    )
    fig = px.scatter(
        scatter_df,
        x="Nom_to_Contest_Ratio",
        y="ContestantsPerSeat",
        color="ConstituencyType",
        hover_data=["State", "FILED", "ACCEPTED", "WITHDRAWN", "CONTESTING", "FORFEITED", "Seats"],
        title="Filed/Contesting Ratio vs. Contestants per Seat (by State & Caste)"
    )
    fig.update_xaxes(title="Filed / Contesting (ratio)")
    fig.update_yaxes(title="Contestants per Seat")
    return fig


def _csv6_sankey_values_for_state(totals: pd.DataFrame, state: str):
    """Computes Sankey flows for one state (or ALL-INDIA)."""
    if state == "ALL-INDIA":
        df = totals[["FILED", "REJECTED", "WITHDRAWN", "CONTESTING", "ACCEPTED"]].sum(numeric_only=True).to_frame().T
    else:
        df = totals[totals["State"] == state][["FILED", "REJECTED", "WITHDRAWN", "CONTESTING", "ACCEPTED"]].sum(numeric_only=True).to_frame().T
    F = float(df["FILED"].iloc[0])
    R = float(df["REJECTED"].iloc[0])
    A = float(df["ACCEPTED"].iloc[0])
    W = float(df["WITHDRAWN"].iloc[0])
    C = float(df["CONTESTING"].iloc[0])
    return F, R, A, W, C


def fig_39_nomination_flow_sankey_dropdown(totals: pd.DataFrame, state_list: list[str]) -> go.Figure:
    """Sankey: FILED → (REJECTED, ACCEPTED) → (WITHDRAWN, CONTESTING) with dropdown to switch state."""
    # Initial ALL-INDIA
    F, R, A, W, C = _csv6_sankey_values_for_state(totals, "ALL-INDIA")
    labels = ["FILED", "REJECTED", "ACCEPTED", "WITHDRAWN", "CONTESTING"]
    node = dict(label=labels, pad=20, thickness=16)
    link = dict(source=[0, 0, 2, 2], target=[1, 2, 3, 4], value=[R, max(F - R, 0.0), min(W, A), max(C, 0.0)])

    fig = go.Figure(go.Sankey(node=node, link=link))
    fig.update_layout(title_text="Nomination Flow — ALL-INDIA", font_size=12, margin=dict(r=160))

    # Buttons: update link values + title
    buttons = []
    for st in state_list:
        F, R, A, W, C = _csv6_sankey_values_for_state(totals, st)
        values = [R, max(F - R, 0.0), min(W, A), max(C, 0.0)]
        buttons.append(dict(
            label=st,
            method="update",
            args=[{"link": [dict(source=[0, 0, 2, 2], target=[1, 2, 3, 4], value=values)]},
                  {"title": f"Nomination Flow — {st}"}]
        ))

    fig.update_layout(updatemenus=[dict(type="dropdown", x=1.15, y=1.0, xanchor="left", buttons=buttons, showactive=True)])
    return fig


def fig_40_contestants_per_seat_heatmap(totals: pd.DataFrame, width: int = 750) -> go.Figure:
    """Clean, compact Heatmap: Contestants per Seat by State × ConstituencyType."""
    # Pivot
    cps = totals.pivot_table(
        index="State",
        columns="ConstituencyType",
        values="ContestantsPerSeat",
        aggfunc="mean"
    )

    # Ensure all three columns exist
    for col in ["GEN", "SC", "ST"]:
        if col not in cps.columns:
            cps[col] = float("nan")

    # Order & sort by mean for nicer ranking
    cps = cps[["GEN", "SC", "ST"]]
    cps["__mean"] = cps.mean(axis=1, skipna=True)
    cps = cps.sort_values("__mean", ascending=False).drop(columns="__mean")

    # Compact height based on number of states
    num_states = len(cps.index)
    height = max(350, num_states * 12 + 120)  # thinner vertical profile

    # Plot
    fig = px.imshow(
        cps,
        labels=dict(x="Constituency Type", y="State", color="Contestants / Seat"),
        x=["GEN", "SC", "ST"],
        y=cps.index,
        title="Contestants per Seat — Heatmap",
        color_continuous_scale="Reds",
        aspect="auto",
    )

    # Style: dark theme, generous bottom margin, no grids
    fig.update_layout(
        height=height,
        width=width,
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=60),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, tickangle=0, tickfont=dict(size=12))
    fig.update_yaxes(showgrid=False, zeroline=False, categoryorder="array", categoryarray=cps.index)

    return fig


def fig_41_rejected_share_top5_states_stacked(totals: pd.DataFrame) -> go.Figure:
    """Stacked bars: Rejected share (Rejected / Filed) by caste for top-5 states by FILED (TOTAL)."""
    top_states = totals.groupby("State")["FILED"].sum().sort_values(ascending=False).head(5).index.tolist()
    stack_df = totals[totals["State"].isin(top_states)].copy()
    stack_df = stack_df.groupby(["State", "ConstituencyType"], as_index=False)[["REJECTED", "FILED"]].sum()
    stack_df["RejectedShare"] = np.where(stack_df["FILED"] > 0, stack_df["REJECTED"] / stack_df["FILED"], 0.0)

    # Nice percent labels
    text_vals = (stack_df["RejectedShare"] * 100.0).round(1).astype(str) + "%"

    fig = px.bar(
        stack_df, x="State", y="RejectedShare", color="ConstituencyType",
        barmode="stack", text=text_vals,
        title="Rejected Share (Rejected / Filed) — Top-5 States by Filed"
    )
    fig.update_yaxes(tickformat=".0%", title="Rejected Share")
    return fig


_CT_ORDER = ["GEN", "SC", "ST"]

def _ensure_forfeiture_rate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ForfeitureRate" not in df.columns:
        df["ForfeitureRate"] = np.where(
            (df.get("CONTESTING", 0) > 0),
            df.get("FORFEITED", 0) / df.get("CONTESTING", 1),
            0.0
        )
    return df


def fig_42_state_leaderboard_metric_by_caste(
    totals: pd.DataFrame,
    metric: str = "RejectionRate",
    constituency_type: str = "GEN",
    metrics_label_map: dict | None = None
) -> go.Figure:
    """
    Horizontal leaderboard of states by a chosen metric for a chosen constituency type.
    metric ∈ {RejectionRate, WithdrawnRate, ForfeitureRate, ContestantsPerSeat, ContestRatio}
    """
    label_map = metrics_label_map or {
        "RejectionRate": "Rejected / Filed",
        "WithdrawnRate": "Withdrawn / Accepted",
        "ForfeitureRate": "Forfeited / Contesting",
        "ContestantsPerSeat": "Contestants per Seat",
        "ContestRatio": "Contesting / Filed",
    }
    base = _ensure_forfeiture_rate(totals)
    base["ConstituencyType"] = pd.Categorical(base["ConstituencyType"], _CT_ORDER, ordered=True)
    df = base[base["ConstituencyType"] == constituency_type].groupby("State", as_index=False)[metric].mean()
    df = df.sort_values(metric, ascending=False)

    text = None
    if "Rate" in metric or "Ratio" in metric:
        text = df[metric].map(lambda v: f"{v:.1%}")

    fig = px.bar(
        df, x=metric, y="State", orientation="h",
        title=f"State Leaderboard — {label_map.get(metric, metric)} ({constituency_type})",
        text=text, template="plotly_dark"
    )
    if "Rate" in metric or "Ratio" in metric:
        fig.update_xaxes(tickformat=".0%")
    fig.update_layout(margin=dict(r=40))
    return fig


def fig_43_dumbbell_filed_to_contesting_by_caste(totals: pd.DataFrame, constituency_type: str = "GEN") -> go.Figure:
    """
    Dumbbell plot comparing FILED vs CONTESTING per state for one constituency type.
    """
    d = totals[totals["ConstituencyType"] == constituency_type].groupby("State", as_index=False)[["FILED", "CONTESTING"]].sum()
    d = d.sort_values("FILED", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["FILED"], y=d["State"], mode="markers", name="FILED",
                             marker=dict(size=8),
                             hovertemplate="State=%{y}<br>Filed=%{x}<extra></extra>"))
    fig.add_trace(go.Scatter(x=d["CONTESTING"], y=d["State"], mode="markers", name="CONTESTING",
                             marker=dict(size=8),
                             hovertemplate="State=%{y}<br>Contesting=%{x}<extra></extra>"))
    for _, row in d.iterrows():
        fig.add_shape(type="line", x0=row["FILED"], y0=row["State"], x1=row["CONTESTING"], y1=row["State"], line=dict(width=2))

    fig.update_layout(
        title=f"Filed → Contesting (Dumbbell) — {constituency_type}",
        xaxis_title="Count", yaxis_title="State", legend_title="Stage",
        height=900, template="plotly_dark"
    )
    return fig


def fig_44_impact_scatter_rejection_vs_forfeiture(totals: pd.DataFrame) -> go.Figure:
    """
    Scatter of RejectionRate vs ForfeitureRate, bubble size = FILED, colored by constituency type.
    """
    impact = _ensure_forfeiture_rate(totals)
    fig = px.scatter(
        impact, x="RejectionRate", y="ForfeitureRate",
        size="FILED", color="ConstituencyType",
        hover_data=["State", "FILED", "REJECTED", "WITHDRAWN", "CONTESTING", "FORFEITED"],
        title="Impact Scatter: Rejection vs Forfeiture (bubble = Filed)", template="plotly_dark"
    )
    fig.update_xaxes(tickformat=".0%", title="Rejection Rate")
    fig.update_yaxes(tickformat=".0%", title="Forfeiture Rate")
    return fig


def fig_45_treemap_share_filed_color_rejectionrate(totals: pd.DataFrame) -> go.Figure:
    """
    Marimekko-like Treemap showing share of FILED by (ConstituencyType → State),
    colored by RejectionRate.
    """
    treemap_df = totals.copy()
    fig = px.treemap(
        treemap_df,
        path=["ConstituencyType", "State"],
        values="FILED",
        color="RejectionRate",
        color_continuous_scale="Reds",
        title="Share of Filed by State & Caste (color = Rejection Rate)",
        template="plotly_dark"
    )
    fig.update_coloraxes(colorbar_title="Rejection Rate")
    return fig


def fig_46_gender_gap_contesting_bar_by_caste(gender_long: pd.DataFrame, constituency_type: str = "GEN") -> go.Figure:
    """
    Gender gap in Contesting: (M − F) / TOTAL by state for one constituency type.
    """
    g_con = gender_long[(gender_long["Metric"] == "CONTESTING") & (gender_long["Gender"].isin(["M", "F", "TOTAL"]))].copy()
    pvt = g_con.pivot_table(
        index=["State", "ConstituencyType"], columns="Gender", values="Value", aggfunc="sum", fill_value=0
    ).reset_index()
    pvt["GenderGapIndex"] = np.where(pvt["TOTAL"] > 0, (pvt["M"] - pvt["F"]) / pvt["TOTAL"], 0.0)

    df = pvt[pvt["ConstituencyType"] == constituency_type].sort_values("GenderGapIndex", ascending=False)
    fig = px.bar(
        df, x="GenderGapIndex", y="State", orientation="h",
        title=f"Gender Gap in Contesting — (M − F) / TOTAL ({constituency_type})",
        text=df["GenderGapIndex"].map(lambda v: f"{v:.1%}"),
        template="plotly_dark"
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_layout(margin=dict(r=40))
    return fig


def fig_47_pareto_forfeitures_by_state(totals: pd.DataFrame) -> go.Figure:
    """
    Pareto showing which states contribute most to FORFEITED counts.
    Left Y: Forfeited count, Right Y: cumulative share.
    """
    pareto = totals.groupby("State", as_index=False)[["FORFEITED"]].sum().sort_values("FORFEITED", ascending=False)
    pareto["CumShare"] = pareto["FORFEITED"].cumsum() / pareto["FORFEITED"].sum()

    fig = go.Figure()
    fig.add_bar(x=pareto["State"], y=pareto["FORFEITED"], name="Forfeited")
    fig.add_trace(go.Scatter(x=pareto["State"], y=pareto["CumShare"], mode="lines+markers",
                             name="Cumulative Share", yaxis="y2"))
    fig.update_layout(
        title="Pareto of Forfeitures by State",
        xaxis_title="State",
        yaxis=dict(title="Forfeited (count)"),
        yaxis2=dict(title="Cumulative Share", overlaying="y", side="right", tickformat=".0%"),
        showlegend=True, template="plotly_dark"
    )
    return fig


def fig_48_withdrawn_rate_strip_by_caste(totals: pd.DataFrame) -> go.Figure:
    """
    Strip plot of WithdrawnRate across states by constituency type.
    Enlarged marker size for better visibility.
    """
    fig = px.strip(
        totals,
        x="ConstituencyType",
        y="WithdrawnRate",
        color="ConstituencyType",
        hover_data=["State", "FILED", "ACCEPTED", "WITHDRAWN"],
        title="Withdrawn Rate Distribution across States (by Caste)",
        template="plotly_dark"
    )

    # Make markers larger and slightly transparent for overlapping clarity
    fig.update_traces(
        marker=dict(size=10, opacity=0.7, line=dict(width=0)),
        jitter=0.25  # spreads points a little for readability
    )

    fig.update_yaxes(
        tickformat=".0%",
        title="Withdrawn / Accepted"
    )
    fig.update_xaxes(title="Constituency Type")
    fig.update_layout(
        margin=dict(l=60, r=40, t=60, b=40)
    )

    return fig


def load_pc_level_csv7(csv_path: str = "7.csv"):
    """
    Loads 7.csv and returns:
      df         : PC-level dataframe with derived columns
      state_agg  : State-level aggregates with weighted metrics
      top_states : Top 12 states by Electors (for crowded viz filters)
    """
    df = pd.read_csv(csv_path)

    # Ensure numeric
    num_cols = [
        "PC No","No Of AC Segments","PS","Total Electors",
        "Avg. No. of Electors Per PS","Nominations","Contestants","FD",
        "Total Voters","VTR (%)"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derived metrics (PC level)
    df["VTR_frac"] = df["VTR (%)"] / 100.0
    df["ElectorsPerPS_calc"] = np.where(df["PS"]>0, df["Total Electors"]/df["PS"], np.nan)
    df["VotersPerPS"]         = np.where(df["PS"]>0, df["Total Voters"]/df["PS"], np.nan)
    df["ContestantsPerPS"]   = np.where(df["PS"]>0, df["Contestants"]/df["PS"], np.nan)
    df["NomsPerPS"]          = np.where(df["PS"]>0, df["Nominations"]/df["PS"], np.nan)
    df["FD_Rate"]            = np.where(df["Contestants"]>0, df["FD"]/df["Contestants"], np.nan)
    df["Nom_to_Cont_Ratio"]   = np.where(df["Contestants"]>0, df["Nominations"]/df["Contestants"], np.nan)
    df["VoterElectorGap"]    = df["Total Electors"] - df["Total Voters"]
    df["GapRate"]            = 1.0 - df["VTR_frac"]
    df["ContPerAC"]          = np.where(df["No Of AC Segments"]>0, df["Contestants"]/df["No Of AC Segments"], np.nan)

    # State aggregates
    state_agg = df.groupby("State", as_index=False).agg(
        PCs=("PC No","count"),
        PS_total=("PS","sum"),
        Electors=("Total Electors","sum"),
        Voters=("Total Voters","sum"),
        Nominations=("Nominations","sum"),
        Contestants=("Contestants","sum"),
        FD=("FD","sum"),
    )
    state_agg["WeightedVTR"] = np.where(state_agg["Electors"]>0, state_agg["Voters"]/state_agg["Electors"], np.nan)
    state_agg["FD_Rate_w"]   = np.where(state_agg["Contestants"]>0, state_agg["FD"]/state_agg["Contestants"], np.nan)
    state_agg["ElectorsPerPS_w"] = np.where(state_agg["PS_total"]>0, state_agg["Electors"]/state_agg["PS_total"], np.nan)
    state_agg["VotersPerPS_w"]   = np.where(state_agg["PS_total"]>0, state_agg["Voters"]/state_agg["PS_total"], np.nan)
    state_agg["Nom_to_Cont_Ratio_w"] = np.where(state_agg["Contestants"]>0, state_agg["Nominations"]/state_agg["Contestants"], np.nan)

    top_states = (
        state_agg.sort_values("Electors", ascending=False)
                 .head(12)["State"].tolist()
    )
    return df, state_agg, top_states


def fig_49_state_weighted_vtr_leaderboard_bar(state_agg: pd.DataFrame) -> go.Figure:
    rank = state_agg.sort_values("WeightedVTR", ascending=False).copy()
    fig = px.bar(
        rank, x="WeightedVTR", y="State", orientation="h",
        text=rank["WeightedVTR"].map(lambda v: f"{v:.1%}"),
        title="Turnout Leaderboard — Weighted by Electors (State)",
        template="plotly_dark"
    )
    fig.update_xaxes(tickformat=".0%", title="Weighted VTR")
    fig.update_layout(yaxis_title="State")
    return fig


def fig_50_pc_vtr_distributions_top12_box(df: pd.DataFrame, top_states_by_electors: list[str]) -> go.Figure:
    box_df = df[df["State"].isin(top_states_by_electors)].copy()
    fig = px.box(
        box_df, x="State", y="VTR_frac", points="suspectedoutliers",
        title="PC-level Turnout Distributions — Top-12 States by Electors",
        template="plotly_dark"
    )
    fig.update_yaxes(tickformat=".0%", title="VTR (PC-level)")
    fig.update_layout(xaxis_title="State")
    return fig


def fig_51_ps_efficiency_electors_vs_voters_scatter(state_agg: pd.DataFrame) -> go.Figure:
    eff = state_agg.copy()
    fig = px.scatter(
        eff, x="ElectorsPerPS_w", y="VotersPerPS_w",
        size="Electors", color="WeightedVTR",
        hover_data=["State","PS_total","Electors","Voters","WeightedVTR","FD_Rate_w"],
        title="Polling Station Efficiency — Electors/PS vs Voters/PS (bubble = Electors, color = VTR)",
        template="plotly_dark"
    )
    fig.update_xaxes(title="Electors per PS (weighted)")
    fig.update_yaxes(title="Voters per PS (weighted)")
    # y = x reference line
    x_min, x_max = eff["ElectorsPerPS_w"].min(), eff["ElectorsPerPS_w"].max()
    fig.add_shape(type="line", x0=x_min, y0=x_min, x1=x_max, y1=x_max, line=dict(width=1, dash="dot"))
    return fig


def fig_52_pc_contestants_per_ps_vs_vtr_scatter(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df, x="ContestantsPerPS", y="VTR_frac",
        color="State", size="Total Electors",
        hover_data=["PC Name","PC No","Contestants","PS","FD","FD_Rate","Nominations"],
        title="Contestants per PS vs Turnout (PC-level)",
        template="plotly_dark"
    )
    fig.update_yaxes(tickformat=".0%", title="VTR")
    fig.update_xaxes(title="Contestants per PS")
    return fig


def fig_53_pc_fd_rate_vs_vtr_scatter(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df, x="VTR_frac", y="FD_Rate",
        color="State", size="Contestants",
        hover_data=["PC Name","Contestants","FD","Total Electors","Total Voters"],
        title="Deposit Forfeiture Rate vs Turnout (PC-level)",
        template="plotly_dark"
    )
    fig.update_xaxes(tickformat=".0%", title="VTR")
    fig.update_yaxes(tickformat=".0%", title="FD / Contestants")
    return fig


def fig_54_state_funnel_nominations_contestants_fd_dropdown(state_agg: pd.DataFrame) -> go.Figure:
    steps = ["Nominations","Contestants","FD"]
    fun_frames = {st: [row[s] for s in steps] for st, row in state_agg.set_index("State").iterrows()}
    init_state = state_agg.sort_values("Electors", ascending=False)["State"].iloc[0]

    fig = go.Figure(go.Funnel(y=steps, x=fun_frames[init_state], textinfo="value+percent previous"))
    buttons = []
    for st in state_agg["State"]:
        buttons.append(dict(
            label=st, method="update",
            args=[{"x":[fun_frames[st]]}, {"title": f"State Funnel — {st}"}]
        ))
    fig.update_layout(
        title=f"State Funnel",
        updatemenus=[dict(type="dropdown", x=1.12, y=1.0, xanchor="left", buttons=buttons, showactive=True)],
        margin=dict(r=160), template="plotly_dark"
    )
    return fig


def fig_55_treemap_electors_state_pc_color_vtr(df: pd.DataFrame) -> go.Figure:
    fig = px.treemap(
        df,
        path=["State","PC Name"],
        values="Total Electors",
        color="VTR_frac",
        color_continuous_scale="Blues",
        title="Electors Composition — State → PC (color = VTR)",
        template="plotly_dark"
    )
    fig.update_coloraxes(colorbar_title="VTR")
    return fig


def fig_56_pareto_fd_by_state(state_agg: pd.DataFrame) -> go.Figure:
    pareto = state_agg.sort_values("FD", ascending=False).copy()
    pareto["CumShare"] = pareto["FD"].cumsum() / pareto["FD"].sum()
    fig = go.Figure()
    fig.add_bar(x=pareto["State"], y=pareto["FD"], name="FD (count)")
    fig.add_trace(go.Scatter(x=pareto["State"], y=pareto["CumShare"], mode="lines+markers",
                             yaxis="y2", name="Cumulative Share"))
    fig.update_layout(
        title="Pareto of Deposit Forfeitures (State)",
        yaxis=dict(title="FD (count)"),
        yaxis2=dict(title="Cumulative Share", overlaying="y", side="right", tickformat=".0%"),
        template="plotly_dark"
    )
    return fig


def fig_57_state_metric_zscore_heatmap(state_agg: pd.DataFrame) -> go.Figure:
    finger = state_agg[[
        "State","WeightedVTR","FD_Rate_w","ElectorsPerPS_w","VotersPerPS_w","Nom_to_Cont_Ratio_w"
    ]].set_index("State").copy()
    z = (finger - finger.mean())/finger.std(ddof=0)

    fig = px.imshow(
        z, labels=dict(x="Metric (z-score)", y="State", color="z"),
        title="State Metric Fingerprint (standardized)",
        aspect="auto", template="plotly_dark"
    )
    # Improve axis tick labels (static positions 0..4 since px.imshow encodes column order)
    fig.update_xaxes(
        ticktext=["WeightedVTR","FD_Rate","Electors/PS","Voters/PS","Nom/Cont Ratio"],
        tickvals=[0,1,2,3,4]
    )
    return fig


def fig_58_dumbbell_electors_vs_voters_by_state(state_agg: pd.DataFrame) -> go.Figure:
    db = state_agg.sort_values("Electors", ascending=True).copy()
    y = db["State"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=db["Electors"], y=y, mode="markers", name="Electors",
        hovertemplate="State=%{y}<br>Electors=%{x:,}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=db["Voters"], y=y, mode="markers", name="Voters",
        hovertemplate="State=%{y}<br>Voters=%{x:,}<extra></extra>"
    ))
    for _, r in db.iterrows():
        fig.add_shape(type="line", x0=r["Electors"], y0=r["State"], x1=r["Voters"], y1=r["State"], line=dict(width=2))
    fig.update_layout(
        title="Dumbbell: Electors → Voters (Turnout Gap) — by State",
        xaxis_title="Count", yaxis_title="State", height=900, template="plotly_dark"
    )
    return fig


def fig_59_nominations_vs_vtr_pc_colored_fd_rate(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df, x="Nominations", y="VTR_frac",
        color="FD_Rate", size="Total Electors",
        color_continuous_scale="Turbo",
        hover_data=["State","PC Name","Contestants","FD","PS"],
        title="Do more nominations correlate with turnout? (PC-level, color = FD Rate)",
        template="plotly_dark"
    )
    fig.update_yaxes(tickformat=".0%", title="VTR")
    return fig


def fig_60_violin_turnout_top6_bottom6(df: pd.DataFrame, state_agg: pd.DataFrame) -> go.Figure:
    rank = state_agg.sort_values("WeightedVTR", ascending=False).copy()
    top6 = rank.head(6)["State"].tolist()
    bot6 = rank.tail(6)["State"].tolist()
    dist_df = df[df["State"].isin(top6 + bot6)].copy()
    dist_df["Band"] = np.where(dist_df["State"].isin(top6), "Top-6 VTR States", "Bottom-6 VTR States")
    fig = px.violin(
        dist_df, x="Band", y="VTR_frac", color="Band", box=True, points="outliers",
        hover_data=["State","PC Name"],
        title="Turnout Distributions: Top-6 vs Bottom-6 States (PC-level)",
        template="plotly_dark"
    )
    fig.update_yaxes(tickformat=".0%", title="VTR")
    return fig

# CSV "7.csv"

# -- Helper: attach India state centroids (approx) for scatter_geo maps --
def _india_state_centroids_df() -> pd.DataFrame:
    centroids = {
        "Andhra Pradesh": (15.9, 79.7),
        "Arunachal Pradesh": (28.0, 94.7),
        "Assam": (26.0, 92.9),
        "Bihar": (25.9, 85.8),
        "Goa": (15.3, 74.1),
        "Gujarat": (22.7, 71.6),
        "Haryana": (29.0, 76.0),
        "Himachal Pradesh": (31.8, 77.1),
        "Karnataka": (15.3, 76.3),
        "Kerala": (10.3, 76.3),
        "Madhya Pradesh": (23.5, 78.7),
        "Maharashtra": (19.7, 75.7),
        "Manipur": (24.7, 93.9),
        "Meghalaya": (25.5, 91.3),
        "Mizoram": (23.2, 92.8),
        "Nagaland": (26.1, 94.4),
        "Odisha": (20.3, 84.6),
        "Punjab": (31.1, 75.4),
        "Rajasthan": (26.9, 73.7),
        "Sikkim": (27.5, 88.5),
        "Tamil Nadu": (11.1, 78.7),
        "Tripura": (23.8, 91.3),
        "Uttar Pradesh": (26.8, 80.9),
        "West Bengal": (23.5, 87.2),
        "Chhattisgarh": (21.3, 82.0),
        "Jharkhand": (23.7, 86.1),
        "Uttarakhand": (30.1, 79.0),
        "Telangana": (17.9, 79.6),
        "Andaman & Nicobar Islands": (11.6, 92.7),
        "Chandigarh": (30.7, 76.8),
        "Dadra & Nagar Haveli and Daman & Diu": (20.3, 73.0),
        "NCT OF Delhi": (28.6, 77.2),
        "Lakshadweep": (10.0, 72.6),
        "Puducherry": (11.9, 79.8),
        "Jammu and Kashmir": (33.5, 75.0),
        "Ladakh": (34.3, 77.6),
    }
    return pd.DataFrame([{"State": s, "lat": lat, "lon": lon} for s, (lat, lon) in centroids.items()])

def attach_india_centroids(state_agg: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of state_agg with lat/lon columns merged in."""
    cent = _india_state_centroids_df()
    out = state_agg.merge(cent, on="State", how="left")
    return out


def fig_61_map_nom_to_cont_ratio_scatter(state_agg_with_geo: pd.DataFrame) -> go.Figure:
    """India scatter map — Nominated/Contestant Ratio (weighted) at state level."""
    fig = px.scatter_geo(
        state_agg_with_geo, lat="lat", lon="lon",
        size="Nom_to_Cont_Ratio_w", color="Nom_to_Cont_Ratio_w",
        hover_data=["State","Nominations","Contestants","Nom_to_Cont_Ratio_w","PCs"],
        color_continuous_scale="Turbo",
        projection="natural earth", scope="asia",
        title="India Scatter Map — Nominated/Contestant Ratio (state-level, weighted)",
        template="plotly_dark"
    )
    fig.update_layout(geo=dict(
        center=dict(lat=22.5, lon=79.5), lataxis_range=[6,36], lonaxis_range=[68,98]
    ))
    return fig


def fig_62_map_raw_contestants_scatter(state_agg_with_geo: pd.DataFrame) -> go.Figure:
    """India scatter map — Raw contestants (state totals)."""
    fig = px.scatter_geo(
        state_agg_with_geo, lat="lat", lon="lon",
        size="Contestants", color="Contestants",
        hover_data=["State","Contestants","Nominations","PCs"],
        color_continuous_scale="Blues",
        projection="natural earth", scope="asia",
        title="India Scatter Map — Raw Number of Contestants (state totals)",
        template="plotly_dark"
    )
    fig.update_layout(geo=dict(
        center=dict(lat=22.5, lon=79.5), lataxis_range=[6,36], lonaxis_range=[68,98]
    ))
    return fig


def fig_63_map_weighted_vtr_scatter(state_agg_with_geo: pd.DataFrame) -> go.Figure:
    """India scatter map — Weighted VTR (bubble = Electors)."""
    fig = px.scatter_geo(
        state_agg_with_geo, lat="lat", lon="lon",
        size="Electors", color="WeightedVTR",
        hover_data=["State","WeightedVTR","Electors","Voters","PCs"],
        color_continuous_scale="Greens",
        projection="natural earth", scope="asia",
        title="India Scatter Map — Weighted Turnout (bubble ~ Electors)",
        template="plotly_dark"
    )
    fig.update_layout(geo=dict(
        center=dict(lat=22.5, lon=79.5), lataxis_range=[6,36], lonaxis_range=[68,98]
    ))
    return fig


def fig_64_pc_top_fd_bar(df7: pd.DataFrame, topN: int = 30) -> go.Figure:
    """Top-N PCs by FD (count)."""
    top_fd = df7.sort_values("FD", ascending=False).head(topN)
    fig = px.bar(
        top_fd, x="FD", y="PC Name", orientation="h",
        color="State",
        title=f"Top {topN} PCs by Deposit Forfeitures (count)",
        hover_data=["State","Contestants","Nominations","VTR (%)"],
        template="plotly_dark"
    )
    return fig


def fig_65_pc_top_fd_rate_bar(df7: pd.DataFrame, min_cont: int = 10, topN: int = 30) -> go.Figure:
    """Top-N PCs by FD Rate (filter min contestants to avoid tiny denominators)."""
    d = df7[df7["Contestants"] >= min_cont].copy()
    d = d.sort_values("FD_Rate", ascending=False).head(topN)
    fig = px.bar(
        d, x="FD_Rate", y="PC Name", orientation="h",
        color="State",
        title=f"Top {topN} PCs by FD Rate (min {min_cont} contestants)",
        hover_data=["State","Contestants","FD","Nominations","VTR (%)"],
        template="plotly_dark"
    )
    fig.update_xaxes(tickformat=".0%")
    return fig


def fig_66_pc_contestants_vs_fd_scatter(df7: pd.DataFrame) -> go.Figure:
    """PC-level relationship: Contestants vs FD (bubble = Electors)."""
    fig = px.scatter(
        df7, x="Contestants", y="FD",
        color="State", size="Total Electors",
        hover_data=["PC Name","VTR (%)","Nominations","FD_Rate"],
        title="Contestants vs FD (PC-level)",
        template="plotly_dark"
    )
    return fig


def fig_67_hist_nonforfeiting_candidates(df7: pd.DataFrame) -> go.Figure:
    """
    Histogram of Non-Forfeiting candidates = Contestants − FD (PC-level),
    annotated with quick inference stats.
    """
    # Ensure NonForfeiting is calculated if not present
    if "NonForfeiting" not in df7.columns:
        df7 = df7.copy()
        df7["NonForfeiting"] = df7["Contestants"] - df7["FD"]
        
    share_majority_forfeit = (df7["FD"] > df7["NonForfeiting"]).mean()
    very_low_nonforf_share = (df7["NonForfeiting"] <= 2).mean()

    fig = px.histogram(
        df7, x="NonForfeiting", nbins=30, color="State",
        title="Histogram — Non-Forfeiting Candidates = Contestants − FD (PC-level)",
        marginal="box", template="plotly_dark"
    )
    fig.add_annotation(
        text=f"Share PCs majority forfeit: {share_majority_forfeit:.1%}",
        xref="paper", yref="paper", x=0.02, y=0.98, showarrow=False,
        bgcolor="rgba(255,255,255,0.7)"
    )
    fig.add_annotation(
        text=f"PCs with ≤2 non-forfeiting: {very_low_nonforf_share:.1%}",
        xref="paper", yref="paper", x=0.02, y=0.92, showarrow=False,
        bgcolor="rgba(255,255,255,0.7)"
    )
    return fig


def fig_68_density_nonforfeiting_vs_vtr(df7: pd.DataFrame) -> go.Figure:
    """2D density: Non-Forfeiting vs VTR (PC-level)."""
    # Ensure NonForfeiting is calculated if not present
    if "NonForfeiting" not in df7.columns:
        df7 = df7.copy()
        df7["NonForfeiting"] = df7["Contestants"] - df7["FD"]
        
    fig = px.density_heatmap(
        df7, x="NonForfeiting", y="VTR_frac",
        nbinsx=30, nbinsy=20, color_continuous_scale="Viridis",
        title="Density — Non-Forfeiting Candidates vs Turnout (PC-level)",
        template="plotly_dark"
    )
    fig.update_yaxes(tickformat=".0%", title="VTR")
    return fig

# CSV "8.csv" — Candidate distribution by state/UT

def load_candidate_distribution_csv8(path: str = "8.csv"):
    """
    Load 8.csv and return:
      - df: cleaned wide table (per State/UT)
      - df_long: melted buckets per State/UT with shares
      - geo_df: df merged with India centroids for mapping
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    bucket_cols = [
        "No of Constituencies With Candidates Numbering >1 < =15",
        "No of Constituencies With Candidates Numbering >15 < =31",
        "No of Constituencies With Candidates Numbering >31 < =47",
        "No of Constituencies With Candidates Numbering >47 < =63",
        "No of Constituencies With Candidates Numbering >63",
    ]
    bucket_labels = ["1–15", "16–31", "32–47", "48–63", "64+"]

    df_long = df.melt(
        id_vars=["State/UT", "No. of Seats"],
        value_vars=bucket_cols,
        var_name="Bucket",
        value_name="Constituencies",
    )
    df_long["Bucket"] = df_long["Bucket"].replace(dict(zip(bucket_cols, bucket_labels)))

    # normalized share within each state
    df_long["Share"] = (
        df_long.groupby("State/UT")["Constituencies"]
        .transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
    )

    # simple India centroids for State/UT
    state_centroids = {
        "Andhra Pradesh": (15.9, 79.7),
        "Arunachal Pradesh": (28.0, 94.7),
        "Assam": (26.0, 92.9),
        "Bihar": (25.9, 85.8),
        "Goa": (15.3, 74.1),
        "Gujarat": (22.7, 71.6),
        "Haryana": (29.0, 76.0),
        "Himachal Pradesh": (31.8, 77.1),
        "Karnataka": (15.3, 76.3),
        "Kerala": (10.3, 76.3),
        "Madhya Pradesh": (23.5, 78.7),
        "Maharashtra": (19.7, 75.7),
        "Manipur": (24.7, 93.9),
        "Meghalaya": (25.5, 91.3),
        "Mizoram": (23.2, 92.8),
        "Nagaland": (26.1, 94.4),
        "Odisha": (20.3, 84.6),
        "Punjab": (31.1, 75.4),
        "Rajasthan": (26.9, 73.7),
        "Sikkim": (27.5, 88.5),
        "Tamil Nadu": (11.1, 78.7),
        "Tripura": (23.8, 91.3),
        "Uttar Pradesh": (26.8, 80.9),
        "West Bengal": (23.5, 87.2),
        "Chhattisgarh": (21.3, 82.0),
        "Jharkhand": (23.7, 86.1),
        "Uttarakhand": (30.1, 79.0),
        "Telangana": (17.9, 79.6),
        "Andaman & Nicobar Islands": (11.6, 92.7),
        "Chandigarh": (30.7, 76.8),
        "Dadra & Nagar Haveli and Daman & Diu": (20.3, 73.0),
        "NCT OF Delhi": (28.6, 77.2),
        "Lakshadweep": (10.0, 72.6),
        "Puducherry": (11.9, 79.8),
        "Jammu and Kashmir": (33.5, 75.0),
        "Ladakh": (34.3, 77.6),
    }
    cent_df = pd.DataFrame(
        [{"State/UT": s, "lat": lat, "lon": lon} for s, (lat, lon) in state_centroids.items()]
    )
    geo_df = df.merge(cent_df, on="State/UT", how="left")

    return df, df_long, geo_df


def fig_69_candidate_bucket_distribution_stackedbar(df_long: pd.DataFrame) -> go.Figure:
    """
    Stacked distribution of constituencies by candidate-count buckets, per State/UT.
    """
    fig = px.bar(
        df_long,
        y="State/UT",
        x="Constituencies",
        color="Bucket",
        orientation="h",
        color_discrete_sequence=px.colors.sequential.Viridis_r,
        title="Distribution of Constituencies by Candidate Count Range (per State/UT)",
        template="plotly_dark",
    )
    fig.update_layout(barmode="stack", height=900)
    return fig


def fig_70_avg_candidates_ranking_bar(df: pd.DataFrame) -> go.Figure:
    """
    Ranking of States/UTs by average candidates per constituency.
    """
    df_rank = df.sort_values("Avg Candidates In a Constituency", ascending=False)
    fig = px.bar(
        df_rank,
        x="Avg Candidates In a Constituency",
        y="State/UT",
        color="Avg Candidates In a Constituency",
        orientation="h",
        color_continuous_scale="Plasma",
        title="Ranking of States/UTs by Average Number of Candidates per Constituency",
        template="plotly_dark",
    )
    fig.update_layout(height=900)
    return fig


def fig_71_avg_vs_max_candidates_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Avg vs Max candidates (bubble = total candidates).
    """
    fig = px.scatter(
        df,
        x="Avg Candidates In a Constituency",
        y="Max Candidates In a Constituency",
        size="Total Candidates",
        color="Total Candidates",
        hover_data=["State/UT","No. of Seats","Min Candidates In a Constituency"],
        color_continuous_scale="Turbo",
        title="Avg vs Max Candidates — Bubble ~ Total Candidates",
        template="plotly_dark",
    )
    fig.add_annotation(
        xref="x", yref="y",
        x=df["Avg Candidates In a Constituency"].max()*0.8,
        y=df["Max Candidates In a Constituency"].max()*0.95,
        text="High spread: far above the diagonal",
        showarrow=True, arrowhead=2, bgcolor="rgba(255,255,255,0.7)"
    )
    return fig


def fig_72_total_candidates_bubble_map(geo_df: pd.DataFrame) -> go.Figure:
    """
    India bubble map — bubble size: Total Candidates; color: Avg Candidates.
    """
    fig = px.scatter_geo(
        geo_df,
        lat="lat", lon="lon",
        size="Total Candidates",
        color="Avg Candidates In a Constituency",
        hover_data=["State/UT","No. of Seats","Max Candidates In a Constituency"],
        color_continuous_scale="Magma",
        projection="natural earth", scope="asia",
        title="India Bubble Map — Total Candidates (Bubble Size) vs Avg Candidates (Color)",
        template="plotly_dark",
    )
    fig.update_layout(
        geo=dict(center=dict(lat=22, lon=79), lataxis_range=[6,36], lonaxis_range=[68,98])
    )
    return fig


# CSV "LokSabhaAssetComparison.csv" — Candidate Asset Changes

def clean_asset_comparison_value(s: str | float | None) -> float:
    """Helper to clean 'Rs 1,23,4561 Crore+' style strings to numeric."""
    if s is None or pd.isna(s):
        return np.nan
    
    s = str(s)
    # This regex removes the summary part (e.g., "272 Crore+")
    s_cleaned = re.sub(r'\d+\s*(?:Crore|Lakh|Thousand)\+.*$', '', s, flags=re.IGNORECASE)
    s_cleaned = s_cleaned.replace('Rs', '').replace(',', '').strip()
    
    return pd.to_numeric(s_cleaned, errors='coerce')

def load_asset_comparison_csv(csv_path):
    """
    Loads and cleans asset columns,
    and derives labels.
    Returns:
      - df_assets: Cleaned DataFrame with asset changes.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return pd.DataFrame()

    # Clean asset increase
    df.dropna(subset=['Asset Increase', 'Name (Party)'], inplace=True)
    df['Asset_Increase_Num'] = df['Asset Increase'].apply(clean_asset_comparison_value)
    
    # Clean asset values for 2024 and 2019 (for potential future plots)
    if 'Total Assets in Lok Sabha 2024' in df.columns:
        df['Assets_2024_Num'] = df['Total Assets in Lok Sabha 2024'].apply(clean_asset_comparison_value)
    if 'Total Assets in Lok Sabha 2019' in df.columns:
        df['Assets_2019_Num'] = df['Total Assets in Lok Sabha 2019'].apply(clean_asset_comparison_value)

    # Extract Candidate Name and Party
    df_extracted = df['Name (Party)'].str.extract(r'^(.*?)\s*\((.*?)\)$')
    if df_extracted is not None and not df_extracted.empty:
        df['Candidate_Name'] = df_extracted[0]
        df['Party'] = df_extracted[1]
        df['Name_Party_Label'] = df['Candidate_Name'].fillna('') + ' (' + df['Party'].fillna('') + ')'
    else:
        df['Name_Party_Label'] = df['Name (Party)']

    df_assets = df.dropna(subset=['Asset_Increase_Num', 'Name_Party_Label']).copy()

    return df_assets.reset_index(drop=True)


def fig_73_asset_increase_decrease_bar(df_assets: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Horizontal bar: Top-N asset increases and Top-N asset decreases (losses).
    Uses a transformed 'symlog'-style axis to handle large value range and negatives.
    """
    if df_assets.empty:
        return go.Figure().update_layout(title=f"Top {top_n} Asset Increases & Decreases (No Data)", template="plotly_dark")
        
    df_sorted = df_assets.sort_values('Asset_Increase_Num', ascending=True)

    df_bottom = df_sorted.head(top_n)
    df_top = df_sorted.tail(top_n)

    df_plot = pd.concat([df_bottom, df_top])

    # Apply Manual 'symlog' Transformation
    df_plot['Asset_Increase_Transformed'] = np.sign(df_plot['Asset_Increase_Num']) * np.log10(np.abs(df_plot['Asset_Increase_Num']) + 1)
    df_plot['Color'] = ['#2ca02c' if x > 0 else '#d62728' for x in df_plot['Asset_Increase_Num']]
    
    df_plot = df_plot.sort_values('Asset_Increase_Num', ascending=True)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_plot['Asset_Increase_Transformed'],
        y=df_plot['Name_Party_Label'],
        customdata=df_plot['Asset_Increase_Num'],
        orientation='h',
        marker_color=df_plot['Color'],
        hovertemplate='<b>%{y}</b><br>Asset Increase: %{customdata:,.0f} Rs<extra></extra>'
    ))

    fig.update_layout(
        title=f'Top {top_n} Asset Increases & Decreases (Transformed Scale)',
        xaxis_title='Asset Increase (Transformed Log Scale)',
        yaxis_title='Candidate (Party)',
        template='plotly_dark',
        xaxis_type='linear',
        
        yaxis={'categoryorder':'array', 'categoryarray': df_plot['Name_Party_Label']},
        
        shapes=[
            go.layout.Shape(
                type="line",
                yref="paper", y0=0, x0=0, y1=1, x1=0, # Vertical line at 0
                line=dict(color="gray", width=2, dash="dot")
            )
        ],
        height=700,
        margin=dict(l=350, b=100) # Left margin for long labels
    )
    return fig