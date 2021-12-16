import pandas as pd
import janitor as jn
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from io import BytesIO

from millify import millify
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_percentage_error, r2_score

st.set_page_config(
    page_title='Model Results',
    page_icon=':100:',
    layout="wide",
)

@st.cache(show_spinner=False, allow_output_mutation=True)
def get_results_data():
    summ_df = pd.read_csv('./data/summary.csv')
    preds_df = pd.read_csv('./data/pred_vs_actual.csv')

    return summ_df, preds_df

@st.cache(show_spinner=False, allow_output_mutation=True)
def get_fam_and_part_num_elasticities(summary_df):
    summary_df['material_number'] = (
        summary_df['index']
        .str
        .extract('log_dealer_price\|material_number\[(\w+)\]')
        .astype(str)
    )

    fam_elasticity = (
        summary_df
        .query("index == 'log_dealer_price'")[['FAMILY', 'mean']]
        .rename(columns={'mean': 'fam_elasticity'})
    )

    part_num_elasticities = (
        summary_df
        .query("material_number != 'nan'")[['FAMILY', 'material_number', 'mean']]
        .rename(columns={'mean': 'pn_elasticity'})
    )

    final_elasticities = pd.merge(
        part_num_elasticities,
        fam_elasticity, 
        on='FAMILY', 
        how='left',
    )

    final_elasticities['total_elasticity'] = -(final_elasticities['fam_elasticity'] + final_elasticities['pn_elasticity'])

    final_elasticities = (
        final_elasticities[['FAMILY', 'material_number', 'total_elasticity']]
        .rename(columns={
            'total_elasticity': 'Elasticity',
            'FAMILY': 'Family',
            'material_number': 'Part Number'
        })
    )
    
    return fam_elasticity, final_elasticities

@st.cache(show_spinner=False, allow_output_mutation=True)
def get_family_metrics(preds_df, fam_elasticity,):
    results = []
    all_fam = preds_df['family'].unique().tolist()

    for fam in all_fam:
        tmp_preds = preds_df.query('family == @fam')
        mape = mean_absolute_percentage_error(tmp_preds['units'] + 1, tmp_preds['preds']) * 100
        wmape = ((tmp_preds['units'] - tmp_preds['preds']).sum()) * 100 / tmp_preds['units'].sum()
        r2 = r2_score(tmp_preds['units'], tmp_preds['preds']) * 100
        fam_ela = -fam_elasticity.query('FAMILY == @fam')['fam_elasticity'].values[0]
        values = [fam, fam_ela, r2, mape, wmape]
        results.append(values)
    
    result_df = pd.DataFrame(results, columns=['Family', 'Elasticity', 'R2', 'MAPE', 'WMAPE'])

    return result_df


def get_total_part_num_in_family(df):
    rename_cols = {
        'description_pfc': 'Family',
        'material_number': 'Total parts',
    }

    num_parts = (
        df
        .groupby('description_pfc')
        .agg({ 'material_number': 'nunique' })
        .reset_index()
        .rename(columns=rename_cols)
    )

    return num_parts


def get_num_of_parts_in_modelling(df):
    
    rename_cols = {
        'Part Number': 'Parts in Model'
    }

    num_parts = (
        df
        .groupby('Family')
        .agg({ 'Part Number': 'nunique' })
        .reset_index()
        .rename(columns=rename_cols)
    )

    return num_parts


def get_profit_percentage(trans, fin_elast):
    parts_in_model = fin_elast['Part Number'].unique().tolist()

    rename_cols = { 
        'description_pfc': 'Family',
        'percent_profit': '% Profit in model'
    }
    
    total_profit = (
        trans
        .add_column('total_profit', trans['dealer_np_ron'] - trans['amount_in_local_currency_ron'])
        .groupby('description_pfc')['total_profit']
        .sum()
        .reset_index()
    )

    partial_profit = (
        trans
        .query('material_number in @parts_in_model')
        .assign(profit_in_model = lambda x: x['dealer_np_ron'] - x['amount_in_local_currency_ron'])
        .groupby('description_pfc')['profit_in_model']
        .sum()
        .reset_index()
    )

    profit_percentage = pd.merge(partial_profit, total_profit, on='description_pfc', how='left')
    profit_percentage['percent_profit'] = profit_percentage['profit_in_model'] * 100 / profit_percentage['total_profit']

    profit_percentage = profit_percentage.rename(columns=rename_cols).drop(['profit_in_model', 'total_profit'], axis=1)

    return profit_percentage

@st.cache(show_spinner=False, allow_output_mutation=True)
def get_additional_family_metrics(fm, fin_elast):
    comp = (
        pd.read_excel(
            './data/comp_type.xlsx', 
            usecols=[1, 2], 
            names=['Family', 'Competitive Type']
        )
        .transform_column('Family', lambda x: x.strip())
    )

    trans_data = pd.read_csv('./data/reconciled_data_v4.csv')
    total_parts_in_family = get_total_part_num_in_family(df=trans_data)
    parts_in_modelling = get_num_of_parts_in_modelling(df=fin_elast)
    profit_percentage = get_profit_percentage(trans=trans_data, fin_elast=fin_elast)

    fm1 = pd.merge(fm, comp, on='Family', how='left')
    fm2 = pd.merge(fm1, total_parts_in_family, on='Family', how='left')
    fm3 = pd.merge(fm2, parts_in_modelling, on='Family', how='left')
    fm4 = pd.merge(fm3, profit_percentage, on='Family', how='left')

    fm4 = fm4.drop_duplicates().reset_index(drop=True)

    return fm4


def get_actuals_vs_pred_plot(df, fam):
    tmp = df.query('family == @fam')

    fig = make_subplots()

    # Add traces
    fig.add_trace(
        go.Scatter( 
            x=tmp['ym'].values, 
            y=tmp['units'].values, 
            name="Actual",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=tmp['ym'].values, 
            y=tmp['preds'].values, 
            name="Predicted",
        ),
    )


    fig.update_layout(
        # title={
        #     'text': f'Actual vs. Predicted quantities',
        #     'y': 0.85,
        #     'x': 0.5,
        #     # 'font_size': 20,
        # },
        xaxis_title='',
        yaxis_title='Quantity',
    )

    return fig


def get_elasticity_scatterplot(df, y_var):
    fig = px.scatter(
        data_frame=df,
        x='Elasticity',
        y=y_var,
        color='Competitive Type',
        symbol='Competitive Type',
        hover_name='Family',
    )

    fig.update_traces(
        marker={ 'size': 15 }
    )

    return fig


def download_as_excel(df):
    col_names = [{'header': col_name} for col_name in df.columns]

    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    
    # format1 = workbook.add_format({'num_format': '0.000'}) 
    worksheet.set_column(0, df.shape[0] - 1, 15) 
    
    worksheet.add_table(0, 0, df.shape[0], df.shape[1]-1, {
        'columns': col_names,
        # 'style' = option Format as table value and is case sensitive 
        # (look at the exact name into Excel)
        'style': 'Table Style Medium 5'
    })
    
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def get_distribution_plot_of_elasticties(df, fam):
    fig = px.box(
        data_frame=df,
        x='Family',
        y='Elasticity',
        color='Family',
    )

    return fig


if __name__ == "__main__":

    _, title_col, _ = st.columns([2, 2, 1])
    title_col.title('Results from Model')

    # summary_df, preds_df = get_results_data()
    upload1, upload2 = st.columns(2)
    preds_file = upload1.file_uploader("Upload the Prediction file")
    summary_file = upload2.file_uploader("Upload the Summary file")


    if not (preds_file is None or summary_file is None):
        preds_df = pd.read_csv(preds_file)
        summary_df = pd.read_csv(summary_file)

        fam_elast, final_elast = get_fam_and_part_num_elasticities(summary_df=summary_df)

        total_families = final_elast['Family'].nunique()
        total_part_nums = final_elast['Part Number'].nunique()

        fam_metrics = get_family_metrics(preds_df=preds_df, fam_elasticity=fam_elast)
        final_fam_metrics = get_additional_family_metrics(fm=fam_metrics, fin_elast=final_elast)

        _, sep_col0, _ = st.columns([2.5, 2, 1])
        sep_col0.header('Overview')

        _, metric_col1, metric_col2, _ = st.columns([2, 2, 2, 1])
        metric_col1.metric('# Family', total_families)
        metric_col2.metric('# Part Nums', total_part_nums)


        df1, df2 = st.columns([1.2, 3])
        df1.caption('Part Number Elasticities')
        df2.caption('Family Metrics')
        
        df1.dataframe(final_elast)
        df2.dataframe(final_fam_metrics)

        part_num_elasticities = download_as_excel(df=final_elast)
        fam_metrics_excel = download_as_excel(df=final_fam_metrics)

        df1.download_button(
            label="Download as Excel",
            data=part_num_elasticities,
            file_name='part_num_elasticities.xlsx',
        )
        df2.download_button(
            label="Download as Excel",
            data=fam_metrics_excel,
            file_name='family_metrics.xlsx',
        )


        _, sep_col1, _ = st.columns([2, 2, 1])
        sep_col1.header('Scatter plot of family elasticity')

        scatter_filter, scatter_plot = st.columns([1, 3])

        y_options = final_fam_metrics.select_dtypes(include=['int', 'float']).columns.tolist()
        scatter_filter.markdown('#')
        y_variable = scatter_filter.selectbox(
            label='Y Variable',
            options=y_options,
            index=6
        )

        elasticity_scatter = get_elasticity_scatterplot(df=final_fam_metrics, y_var=y_variable)
        scatter_plot.plotly_chart(elasticity_scatter, use_container_width=True)

        _, sep_col2, _ = st.columns([2, 2, 1])
        sep_col2.header('Actual vs. Predicted')

        unique_family_names = preds_df['family'].unique().tolist()

        elasticity_box_plot = get_distribution_plot_of_elasticties(final_elast, fam=unique_family_names)
        st.plotly_chart(elasticity_box_plot, use_container_width=True)

        # _, family_filter, _ = st.columns(3)
        # family_name = family_filter.selectbox('Family Name', unique_family_names, index=0)
        
        # act_pred_plot = get_actuals_vs_pred_plot(df=preds_df, fam=family_name)
        # st.plotly_chart(act_pred_plot, use_container_width=True)

    else:
        st.warning(
            '''Please upload the predicitions file and the 
            summary file and ensure that they are in 
            same format as shown below'''
        )   

        sample1, sample2 = st.columns([1, 2])

        preds_dict = {
            "ym": {
                "0": "2019-07-01",
                "1": "2019-08-01",
                "2": "2019-09-01",
                "3": "2019-10-01",
                "4": "2019-11-01"
            },
            "family": {
                "0": "Engine Oil",
                "1": "Engine Oil",
                "2": "Engine Oil",
                "3": "Engine Oil",
                "4": "Engine Oil"
            },
            "units": {
                "0": 14204,
                "1": 9492,
                "2": 13076,
                "3": 6563,
                "4": 7781
            },
            "preds": {
                "0": 6169.2257148815415,
                "1": 9941.680144786093,
                "2": 8872.66768479531,
                "3": 7667.19063800542,
                "4": 7611.111236252852
            }
        }

        summary_dict = {
            "index": {
                "0": "Intercept",
                "1": "month_name[Aug]",
                "2": "month_name[Dec]",
                "3": "month_name[Feb]",
                "4": "month_name[Jan]"
            },
            "mean": {
                "0": 9.618,
                "1": 0.242,
                "2": 0.346,
                "3": -0.172,
                "4": -0.349
            },
            "sd": {
                "0": 2.165,
                "1": 0.18,
                "2": 0.192,
                "3": 0.217,
                "4": 0.204
            },
            "hdi_3%": {
                "0": 5.858,
                "1": -0.124,
                "2": -0.023,
                "3": -0.489,
                "4": -0.757
            },
            "hdi_97%": {
                "0": 13.113,
                "1": 0.588,
                "2": 0.736,
                "3": 0.27,
                "4": 0.073
            },
            "mcse_mean": {
                "0": 1.28,
                "1": 0.009,
                "2": 0.009,
                "3": 0.079,
                "4": 0.02
            },
            "mcse_sd": {
                "0": 1.072,
                "1": 0.007,
                "2": 0.006,
                "3": 0.058,
                "4": 0.015
            },
            "ess_bulk": {
                "0": 3,
                "1": 302,
                "2": 297,
                "3": 8,
                "4": 68
            },
            "ess_tail": {
                "0": 12,
                "1": 972,
                "2": 711,
                "3": 266,
                "4": 48
            },
            "r_hat": {
                "0": 1.64,
                "1": 1.2,
                "2": 1.11,
                "3": 1.18,
                "4": 1.13
            },
            "FAMILY": {
                "0": "Engine Oil",
                "1": "Engine Oil",
                "2": "Engine Oil",
                "3": "Engine Oil",
                "4": "Engine Oil"
            }
        }

        preds_sample_df = pd.DataFrame.from_dict(preds_dict)
        summ_sample_df = pd.DataFrame.from_dict(summary_dict)

        sample1.subheader('Sample Predictions file')
        sample1.dataframe(preds_sample_df)

        sample2.subheader('Sample Summary file')
        sample2.dataframe(summ_sample_df)
