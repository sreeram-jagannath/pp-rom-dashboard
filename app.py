'''
which parts numbers were there in campaign for a duration?
market basket analysis
top part numbers in family pie chart 

'''

from os import rename
from numpy import number
import pandas as pd
import janitor as jn
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from millify import millify
from datetime import datetime, date

st.set_page_config(layout="wide")

@st.cache(show_spinner=False, allow_output_mutation=True)
def get_data():
    data = pd.read_csv('merged_data.csv') 
    data['quotation_confirmation_date_from_tme'] = pd.to_datetime(data['quotation_confirmation_date_from_tme'], dayfirst=True)
    
    data['per_unit_rrp_ron'] = data['recommended_rp_ron'] / data['quantity']
    data['per_unit_dnp_ron'] = data['dealer_np_ron'] / data['quantity']   
    
    return data

# get list of top part numbers when filtered by a family
def get_top_part_numbers(df, pf):
    top_parts = (
        df
        .query('description_pfc == @pf')
        .groupby(['material_number'])['dealer_np_ron']
        .sum()
        .reset_index(name='total_dealer_np')
        .sort_values(by='total_dealer_np', ascending=False)
        .head(500)['material_number']
    ).values.tolist()

    return top_parts

# demand price dataframe for a particular part number
def get_demand_price_df(df, pn):
    groupby_cols = [
        'material_number',
        'conf_month',
    ]

    agg_map = {
        'quantity': 'sum',
        'description_ro': 'first',
        'description_pfc': 'first',
        'unit_purchase_price_eur': 'mean',
        'recommended_rp_ron': 'sum',
        'dealer_np_ron': 'sum',
    }

    rename_cols = {
        'quantity': 'Quantity',
        'recommended_rp_per_unit_ron': 'MRRP per unit',
        'dealer_np_per_unit_ron': 'DNP per unit',
        'description_ro': 'Part Name',
        'description_pfc': 'Part Family',
    }

    demand_price = (
        df
        .query('material_number == @pn')
        .sort_values(by='quotation_confirmation_date_from_tme')
        .assign(conf_month = lambda x: x['quotation_confirmation_date_from_tme'].dt.strftime('%m-%Y'))
        .groupby(groupby_cols)
        .agg(agg_map)
        .reset_index()
        .assign(
            recommended_rp_per_unit_ron = lambda x: x['recommended_rp_ron'] / x['quantity'],
            dealer_np_per_unit_ron = lambda x: x['dealer_np_ron'] / x['quantity'],
        )
        .change_type('conf_month', 'datetime64')
        .sort_values(by='conf_month')
        .rename(columns=rename_cols)
    )

    return demand_price

# plotly chart from the demand price dataframe
def get_demand_price_chart(df, corr):
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{ "secondary_y": True }]])

    # Add traces
    fig.add_trace(
        go.Scatter( 
            x=df['conf_month'].values, 
            y=df['Quantity'].values, 
            name="Quantity",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df['conf_month'].values, 
            y=df['MRRP per unit'].values, 
            name="MRRP per unit",
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=df['conf_month'].values, 
            y=df['DNP per unit'].values, 
            name="DNP per unit",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title={
            'text': f'Correlation b/w Quantity & Per Unit DNP: {corr}',
            'x': 0.075,
            # 'xanchor': 'center',
            'font_size': 20,
            'font_family': 'Arial',
        },
        xaxis_title='Month',
        yaxis_title='Quantity',
        yaxis2_title='Per Unit Price (RON)',
    )

    return fig

# total part numbers in a part family (metric)
def get_num_parts_in_family(df, pf):
    num_parts = df.query('description_pfc == @pf')['material_number'].nunique()
    return num_parts

# get part name of a part number
def get_part_name_family(df, pn):
    df1 = df.query('material_number == @pn').iloc[0]
    part_name = df1['description_ro']
    part_family = df1['description_pfc']

    return part_name, part_family

# filter the dataframe for the given data range
def get_date_filtered_data(df, date_range):
    mn_date, mx_date = date_range
    filt_df = df.query('@mn_date <= quotation_confirmation_date_from_tme <= @mx_date')
    return filt_df

# get box plot of price distributions for a part family
def get_price_dist_boxplot(df, pf):
    # st.write(df.columns)
    df_filt = (
        df
        .query('description_pfc == @pf')
        .select_columns(['description_pfc', 'per_unit_dnp_ron'])
        .drop_duplicates()
    )
    # st.write(df_filt.shape)

    fig = px.box(df_filt, x='description_pfc', y='per_unit_dnp_ron')
    fig.update_layout(
        title={
            'text': 'Distribution of DNP',
            'x': 0.5,
            'xanchor': 'center',
        },
        yaxis_title='Per Unit Dealer Net Price (RON)',
        xaxis_title='',
        width=450,
        # height=300,
    )
    return fig

# total DNP of a part family (metric)
def get_family_revenue(df, family):
    revenue = df.query('description_pfc == @family')['dealer_np_ron'].sum()
    return revenue

# get the dataframe with corr, cov, total quantity, num of months for a part family
def get_top_parts_df_in_family(df, pn_list):
    parts_corr = []

    for pn in pn_list:
        tmp = get_demand_price_df(df=df, pn=pn)
        corr = get_correlation(df=tmp)        
        
        values = [pn, tmp['Part Name'].unique()[0]]
        values.append(corr)
        
        rrp_cov = tmp['MRRP per unit'].std() * 100 / tmp['MRRP per unit'].mean()
        dnp_cov = tmp['DNP per unit'].std() * 100 / tmp['DNP per unit'].mean()

        # values.append(rrp_cov)
        values.append(dnp_cov)

        values.append(tmp.shape[0])
        values.append(tmp['Quantity'].sum())
        
        values.append(tmp['DNP per unit'].mean())
        
        parts_corr.append(values)
  

    columns = [
        'Part Number',
        'Part Name',
        'Qty - DNP corr.', 
        # 'MRRP COV', 
        'DNP COV',
        'No. of months',
        'Total Quantity',
        'Mean per unit DNP',
        ]

    top_parts_corr = pd.DataFrame(data=parts_corr, columns=columns)
    return top_parts_corr


# def get_top_parts_in_family(df, family):
    # rename_cols = {
    #     'material_number': 'Material Number',
    #     'total_dnp': 'Total DNP (RON)',
    #     'dnp_perc': 'DNP %',
    #     'dnp_cumm_perc': 'Cumulative DNP %',
    #     'total_quantity': 'Total Quantity'
    # }

    # top_parts = (
    #     df
    #     .query('description_pfc == @family')
    #     .groupby('material_number')
    #     .agg(
    #         total_quantity = ('quantity', 'sum'),
    #         num_months = ('quantity', 'count'),
    #         total_dnp = ('dealer_np_ron', 'sum')
    #     )
    #     .reset_index()
    #     .sort_values(by='total_dnp', ascending=False)
    #     .assign(
    #         dnp_perc=lambda x: 100 * x['total_dnp'] / df['dealer_np_ron'].sum(),
    #         dnp_cumm_perc=lambda x: x['dnp_perc'].cumsum(),
    #     )
    #     .reset_index(drop=True)
    #     .query('dnp_cumm_perc <= 60')
    #     .rename(columns=rename_cols)
    # )

    # return top_parts

# frequently transacted parts numbers along with the target part number
def get_frequent_combinations(df, pn, pn_trans):
    rename_cols = {
        'material_number_y': 'Material Number',
        'quotation_number': 'Num of Transactions',
        'description_ro': 'Part Name',
        'description_pfc': 'Family Name',
        'perc_transactions': '% Transactions'
    }

    mn_df = df.select_columns(['quotation_number', 'material_number'])
    comb = (
        pd.merge(mn_df, mn_df, on='quotation_number')
        .groupby(['material_number_x', 'material_number_y'])
        .count()
        .reset_index()
        .sort_values(by='quotation_number', ascending=False)
        .filter_on('material_number_x != material_number_y')
        .query('material_number_x == @pn')
        .rename(columns=rename_cols)
        .assign(
            perc_transactions = lambda x: x['Num of Transactions'] * 100 / pn_trans
        )
        .drop('material_number_x', axis=1)
        .reset_index(drop=True)
    )

    number_name_family = (
        pd.merge(
            comb,
            (
                df
                .groupby(['material_number'])
                .agg({'description_ro': 'first', 'description_pfc': 'first'})
                .reset_index()
            ),
            left_on='Material Number',
            right_on='material_number',
            how='left'
        )
        .drop('material_number', axis=1)
        .rename(columns=rename_cols)
    )

    # number_name_family['% Transactions'] = number_name_family['Num of Transactions'] * 100 / pn_trans


    return number_name_family

# pie chart of part families transacted along with target part number
def get_frequent_pie_chart(df):
    df1 = (
        df
        .groupby('Family Name')['Num of Transactions']
        .sum()
        .reset_index()
        .sort_values(by='Num of Transactions', ascending=False)
        .head(10)
    )
    fig = px.pie(df1, values='Num of Transactions', names='Family Name')
    fig.update_layout(
        title={
            'text': 'Top 10 Frequently bought product families along with this part number',
            'x': 0.5,
            'xanchor': 'center'

        }
    )
    return fig

# number of part family transactins (metric)
def get_num_of_family_transactions(df, family):
    counts = df['description_pfc'].value_counts()[family]
    return counts

# number of part number transactions (metric)
def get_part_num_transactions(df, pn):
    counts = df['material_number'].value_counts()[pn]
    return counts

# total part number dealer net price (metric)
def get_part_number_revenue(df, pn):
    revenue = df.query('material_number == @pn')['dealer_np_ron'].sum()
    return revenue

# total quantity of the part number sold (metric)
def get_part_num_quantity(df, pn):
    total_quant = df.query('material_number == @pn')['quantity'].sum()
    return total_quant

# qty - dnp spearman correlation of a particular part number
def get_correlation(df):
    corr = df[['Quantity', 'DNP per unit']].corr(method='spearman').values[0][1]
    # st.write(corr)
    return f'{corr:.2f}'


def is_in_top_60_perc_family(df, pf):
    pass

if __name__ == "__main__":

    data = get_data()

    min_date = data['quotation_confirmation_date_from_tme'].min().date()
    max_date = data['quotation_confirmation_date_from_tme'].max().date()
    date_range = st.slider('Date Range', min_date, max_date, (min_date, max_date))
    # print(date_range)
    data_filt = get_date_filtered_data(df=data, date_range=date_range)
    # st.write(data_filt.shape)

    unique_pfc = data['description_pfc'].unique().tolist()
    # num_unique_pfc = data_filt['description_pfc'].nunique()

    
    # with st.sidebar:
    _, family_filter, _ = st.columns(3)
    product_family = family_filter.selectbox("Select Family Name:", unique_pfc, index=0)
    
    top_part_numbers = get_top_part_numbers(df=data_filt, pf=product_family)
    # with st.sidebar:
    #     part_number = st.selectbox("Select Part Number:", top_part_numbers, index=0)


    st.markdown(f"<h2 style='text-align: center;'>{product_family}</h2>", unsafe_allow_html=True)

    fam_metric1, fam_metric2, fam_metric3, fam_metric4 = st.columns([1, 1, 1, 1])
    num_parts = get_num_parts_in_family(df=data_filt, pf=product_family)
    fam_metric1.metric('# Parts', num_parts)

    fam_num_trans = get_num_of_family_transactions(df=data_filt, family=product_family)
    fam_metric2.metric('# Transactions', millify(fam_num_trans))

    revenue = get_family_revenue(df=data_filt, family=product_family)
    fam_metric3.metric('Revenue (RON)', millify(revenue))

    perc_rev = revenue * 100 / data_filt['dealer_np_ron'].sum()
    fam_metric4.metric('% Revenue', f'{perc_rev:.2f}')
    
    # top_parts, box_dist = st.columns([1.5, 1])
    # st.dataframe(get_demand_price_df(df=data_filt, pn=part_number))
    top_parts_in_family = get_top_parts_df_in_family(df=data_filt, pn_list=top_part_numbers)
    st.dataframe(top_parts_in_family, height=400)
    
    # Box plot of price distributions in a family
    price_dist = get_price_dist_boxplot(df=data_filt, pf=product_family)
    _, box_dist_cont, _ = st.columns(3)
    box_dist_cont.plotly_chart(price_dist, width=350)

    _, pn_filter, _ = st.columns(3)
    part_number = pn_filter.selectbox("Select Part Number:", top_part_numbers, index=0)

    part_name, part_family = get_part_name_family(df=data, pn=part_number)
    st.markdown(f"<h2 style='text-align: center;'>{part_number}</h2>", unsafe_allow_html=True)

    name, family = st.columns(2)
    name.info(f'Part Name: {part_name}')
    family.warning(f'Part Family: {part_family}')

    pn_metric1, pn_metric2, pn_metric3 = st.columns(3)
    part_num_transactions = get_part_num_transactions(df=data_filt, pn=part_number)
    part_num_revenue = get_part_number_revenue(df=data_filt, pn=part_number)
    part_num_total_quant = get_part_num_quantity(df=data_filt, pn=part_number)

    pn_metric1.metric('# Transactions',  millify(part_num_transactions))
    pn_metric2.metric('Revenue (RON)', millify(part_num_revenue))
    pn_metric3.metric('Total Quantity Sold', millify(part_num_total_quant))

    demand_df = get_demand_price_df(df=data_filt, pn=part_number)
    quant_dnp_corr = get_correlation(demand_df)
    demand_chart = get_demand_price_chart(df=demand_df, corr=quant_dnp_corr)
    st.plotly_chart(demand_chart, use_container_width=True)

    st.markdown(f"<h2 style='text-align: center;'>Market Basket</h2>", unsafe_allow_html=True)
    st.success(f'Which part numbers are transacted together with material number {part_number}?')
    combinations = get_frequent_combinations(data_filt, pn=part_number, pn_trans=part_num_transactions)
    st.dataframe(combinations)
    
    comb_pie = get_frequent_pie_chart(combinations)
    st.plotly_chart(comb_pie, use_container_width=True)
