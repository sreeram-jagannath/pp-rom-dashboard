import pandas as pd
import janitor as jn
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from millify import millify
from plotly.subplots import make_subplots

st.set_page_config(
    page_title='Parts Pricing Romania',
    page_icon=':car:',
    layout="wide",
)

def get_act_vs_pred_sheet(df):
    dtype_map = {
        'ym': 'datetime64', 
        'material_number': str, 
        'Description_PFC': str, 
        'actual quantity': int,
        'predicted quantity': float, 
        'error(actuals - predicted)': float
    }

    df = df.get('actuals vs pred').astype(dtype_map)
    df['material_number'] = df['material_number'].str.split('.').str[0]

    return df


def get_comparison_sheet(df):
    dtype_map = {
        'Material Number': str,
        'Sum of actuals': int,
        'Sum of predicted': float
    }

    df = df.get('comparison')
    df.columns = df.iloc[1, :].values.tolist()
    df = df.iloc[2:-1, :].astype(dtype_map)
    df['Material Number'] = df['Material Number'].str.split('.').str[0]

    return df.reset_index(drop=True)


def get_results_sheet(df):
    dtype_map = {
        'material_number': str, 
        'FINAL_FAMILY': str, 
        'mean': float, 
        'r2_score': float, 
        'wmape': float, 
        'rmse': float,
        'mape': float, 
        'prior_mean': float,
        'prior_std': float,
        'fixed_slope_global_intercept': float,
        'months count where quant sold>0': int, 
        'told quantity sold': int,
    }

    df = df.get('results').astype(dtype_map)

    return df

@st.cache(show_spinner=False, allow_output_mutation=True)
def get_data():
    data = pd.read_excel('./data/model_results.xlsx', sheet_name=None, engine='openpyxl')   

    act_pred_sheet = get_act_vs_pred_sheet(df=data)
    comparison_sheet = get_comparison_sheet(df=data)
    results_sheet = get_results_sheet(df=data)

    return results_sheet, comparison_sheet, act_pred_sheet


def get_actuals_vs_pred_plot(df, pn):
    tmp = df.query('material_number == @pn')

    fig = make_subplots()

    # Add traces
    fig.add_trace(
        go.Scatter( 
            x=tmp['ym'].values, 
            y=tmp['actual quantity'].values, 
            name="Actual",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=tmp['ym'].values, 
            y=tmp['predicted quantity'].values, 
            name="Predicted",
        ),
    )

        # fig.add_trace(
        #     go.Scatter(
        #         x=tmp['ym'].values, 
        #         y=tmp['error(actuals - predicted)'].values, 
        #         name="Error",
        #     ),
        # )

    fig.update_layout(
        title={
            'text': f'Actual vs. Predicted quantities',
            'y': 0.85,
            # 'font_size': 20,
        },
        xaxis_title='',
        yaxis_title='Quantity',
    )

    return fig


def get_family_name_in_comp_sheet(df, act_vs_pred):
    family_map = (
        act_vs_pred
        .groupby('material_number')
        .agg({'Description_PFC': 'first'})
        .to_dict()
        .get('Description_PFC')
    )
    
    df['family_name'] = df['Material Number'].map(family_map)
    return df


def get_part_family(df, pn):
    p_fam = df.query('material_number == @pn')['Description_PFC'].unique().tolist()[0]
    return p_fam


def get_part_numbers_in_family(comp_df, pf):
    pns = comp_df.query('family_name == @pf')['Material Number'].unique().tolist()
    return pns


def get_family_level_plot(pn, act, pred):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=pn,
            y=act,
            name='Actual',
            # marker=dict(
            #     color='rgba(156, 165, 196, 0.95)',
            #     line_color='rgba(156, 165, 196, 1.0)',
            # )
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=pn, 
            y=pred,
            name='Predicted',
            # marker=dict(
            #     color='rgba(204, 204, 204, 0.95)',
            #     line_color='rgba(217, 217, 217, 1.0)'
            # )
        )
    )

    fig.update_traces(
        mode='markers', 
        marker=dict(line_width=1, symbol='circle', size=8))

    fig.update_layout(
        title={
            'text': f'Sum of all quantities',
            'y': 0.85,
            'font_size': 20,
        },
        xaxis_title='',
        yaxis_title='Total Quantity',
    )

    return fig  


def get_family_level_predictions(df, pn_list):
    tmp = df.query('`Material Number` in @pn_list')

    part_numbers = tmp['Material Number'].values.tolist()
    actual_quant = tmp['Sum of actuals'].values.tolist()
    pred_quant = tmp['Sum of predicted'].values.tolist()

    fig = get_family_level_plot(
        pn=part_numbers, 
        act=actual_quant, 
        pred=pred_quant
    )
    return fig


if __name__ == "__main__":

    results, comparison, act_pred = get_data()

    comparison = get_family_name_in_comp_sheet(df=comparison, act_vs_pred=act_pred)

    _, title_text, _ = st.columns(3)
    title_text.title('Results')

    unique_part_numbers = act_pred['material_number'].unique().tolist()
    unique_family_names = comparison['family_name'].unique().tolist()

    _, family_filter, _ = st.columns(3)
    family_name = family_filter.selectbox('Family Name', unique_family_names, index=0)

    pn_list = get_part_numbers_in_family(comp_df=comparison, pf=family_name)
    part_nums_in_fam = st.multiselect(
        label='Part Numbers',
        options=pn_list,
        default=pn_list[:10],
    ) 
    family_level_preds = get_family_level_predictions(df=comparison, pn_list=part_nums_in_fam)
    st.plotly_chart(family_level_preds, use_container_width=True)

    _, pn_filter, _ = st.columns(3)
    part_number = pn_filter.selectbox('Part Number', unique_part_numbers, index=0)

    # part_family = get_part_family(df=df, pn=part_number)
    # st.info(f'Family Name: {part_family}')

    act_pred_plot = get_actuals_vs_pred_plot(df=act_pred, pn=part_number)
    st.plotly_chart(act_pred_plot, use_container_width=True)

