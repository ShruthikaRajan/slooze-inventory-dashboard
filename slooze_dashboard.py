import dash
from dash import dcc, html, dash_table, Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Load datasets
sales = pd.read_csv('slooze_challenge/SalesFINAL12312016.csv', parse_dates=['SalesDate'])
purchase_prices = pd.read_csv('slooze_challenge/2017PurchasePricesDec.csv')
invoice_purchases = pd.read_csv('slooze_challenge/InvoicePurchases12312016.csv')
purchases = pd.read_csv('slooze_challenge/PurchasesFINAL12312016.csv', parse_dates=['PODate', 'ReceivingDate'])

# Preprocessing
sales['Description'] = sales['Description'].astype(str).str.strip().replace(['', 'nan', 'NaN', 'None'], 'Unknown')
purchases['Description'] = purchases['Description'].astype(str).str.strip().replace(['', 'nan', 'NaN', 'None'], 'Unknown')
purchases['VendorName'] = purchases['VendorName'].astype(str).str.strip().replace(['', 'nan', 'NaN', 'None'], 'Unknown')
purchases['Description'] = purchases['Description'].str.title()
sales['Description'] = sales['Description'].str.title()

# Remove duplicates
purchases = purchases.drop_duplicates()
sales = sales.drop_duplicates()
invoice_purchases = invoice_purchases.drop_duplicates()
purchase_prices = purchase_prices.drop_duplicates()

# Brand lookup
brand_lookup = sales[['Brand', 'Description']].drop_duplicates()

# ABC Classification
brand_sales = sales.groupby('Brand')['SalesDollars'].sum().reset_index()
brand_sales = pd.merge(brand_sales, brand_lookup, on='Brand', how='left')
brand_sales = brand_sales.sort_values(by='SalesDollars', ascending=False).reset_index(drop=True)
brand_sales['CumulativeSales'] = brand_sales['SalesDollars'].cumsum()
brand_sales['CumulativePerc'] = 100 * brand_sales['CumulativeSales'] / brand_sales['SalesDollars'].sum()

def abc_category(cum_perc):
    if cum_perc <= 70:
        return 'A'
    elif cum_perc <= 90:
        return 'B'
    else:
        return 'C'

brand_sales['ABC_Category'] = brand_sales['CumulativePerc'].apply(abc_category)

# Lead Time Calculations
purchases['LeadTimeDays'] = (purchases['ReceivingDate'] - purchases['PODate']).dt.days
purchases['LeadTimeDays'] = purchases['LeadTimeDays'].fillna(purchases['LeadTimeDays'].median())
purchases = purchases[purchases['LeadTimeDays'] >= 0]

leadtime_stats_vendor = purchases.groupby('VendorName')['LeadTimeDays'].agg(['mean', 'median', 'std', 'min', 'max', 'count']).reset_index()
leadtime_stats_brand = purchases.groupby('Brand')['LeadTimeDays'].agg(['mean', 'median', 'std', 'min', 'max', 'count']).reset_index()
leadtime_stats_brand = pd.merge(leadtime_stats_brand, brand_lookup, on='Brand', how='left')

# EOQ Calculation
annual_demand = sales.groupby('Brand')['SalesQuantity'].sum().reset_index().rename(columns={'SalesQuantity': 'D'})
merged_invoice = pd.merge(invoice_purchases, purchases[['VendorNumber', 'PONumber', 'Brand']], on=['VendorNumber', 'PONumber'], how='left')
avg_freight_by_brand = merged_invoice.groupby('Brand')['Freight'].mean().reset_index().rename(columns={'Freight': 'OrderingCost'})

annual_demand = pd.merge(annual_demand, avg_freight_by_brand, on='Brand', how='left')
annual_demand = pd.merge(annual_demand, purchase_prices[['Brand', 'PurchasePrice']], on='Brand', how='left')
annual_demand = pd.merge(annual_demand, brand_lookup, on='Brand', how='left')
annual_demand['HoldingCost'] = annual_demand['PurchasePrice'] * 0.2
annual_demand['EOQ'] = np.sqrt((2 * annual_demand['D'] * annual_demand['OrderingCost']) / annual_demand['HoldingCost'])
annual_demand = annual_demand.dropna(subset=['EOQ'])
annual_demand['EOQ'] = annual_demand['EOQ'].round().astype(int)

purchases['YearMonth'] = purchases['PODate'].dt.to_period('M').dt.to_timestamp()
leadtime_monthly_vendor = purchases.groupby(['VendorName', 'YearMonth'])['LeadTimeDays'].mean().reset_index()
leadtime_monthly_vendor['LeadTimeDays'] = leadtime_monthly_vendor['LeadTimeDays'].round(2)

# Demand Forecasting Setup
daily_sales = sales.groupby(['Brand', 'Description', 'SalesDate'])['SalesQuantity'].sum().reset_index()
forecast_horizon = 30

def moving_average_forecast(df, window=7, forecast_days=30):
    df = df.sort_values('SalesDate')
    df['MA_Sales'] = df['SalesQuantity'].rolling(window=window, min_periods=1).mean()
    last_date = df['SalesDate'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    last_ma = df['MA_Sales'].iloc[-1] if not df['MA_Sales'].empty else 0
    forecast_df = pd.DataFrame({'SalesDate': future_dates, 'ForecastQuantity': last_ma})
    return forecast_df

# Dash App
app = dash.Dash(__name__)
app.title = "Slooze Retail Inventory Dashboard"

app.layout = html.Div([
    html.H1("Slooze Retail Inventory Dashboard", style={'textAlign': 'center'}),

    # ABC Analysis Section
    html.Div([
        html.H3("ABC Inventory Classification"),
        dcc.Dropdown(
            id='abc-visual-dropdown',
            options=[
                {'label': 'Bar Chart', 'value': 'bar'},
                {'label': 'Pie Chart', 'value': 'pie'},
                {'label': 'Box Plot', 'value': 'box'}
            ],
            value='bar',
            clearable=False,
            style={'width': '40%', 'marginBottom': '10px'}
        ),
        dcc.Graph(id='abc-bar-chart'),
    ], style={'backgroundColor': 'white', 'padding': '10px', 'margin': '10px'}),

    # Lead Time by Vendor
    html.Div([
        html.H3("Lead Time Statistics by Vendor"),
        html.Button("Export Vendor Lead Time CSV", id='download-vendor-btn', n_clicks=0, style={
            'marginBottom': '10px', 'backgroundColor': '#17a2b8', 'color': 'white', 'padding': '10px',
            'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'
        }),
        dcc.Download(id="download-vendor-data"),
        dash_table.DataTable(
            id='leadtime-table-vendor',
            columns=[{'name': col, 'id': col} for col in leadtime_stats_vendor.columns],
            data=[],
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center'},
            style_header={'fontWeight': 'bold'}
        )
    ], style={'backgroundColor': 'white', 'padding': '10px', 'margin': '10px'}),

    # Lead Time by Brand
    html.Div([
        html.H3("Lead Time Statistics by Brand"),
        html.Button("Export Brand Lead Time CSV", id='download-brand-btn', n_clicks=0, style={
            'marginBottom': '10px', 'backgroundColor': '#ffc107', 'color': 'black', 'padding': '10px',
            'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'
        }),
        dcc.Download(id="download-brand-data"),
        dash_table.DataTable(
            id='leadtime-table-brand',
            columns=[{'name': col, 'id': col} for col in leadtime_stats_brand.columns],
            data=[],
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center'},
            style_header={'fontWeight': 'bold'}
        )
    ], style={'backgroundColor': 'white', 'padding': '10px', 'margin': '10px'}),

    # Lead Time Trend Line
    html.Div([
        html.H3("Lead Time Trend Over Time by Vendor"),
        dcc.Dropdown(
            id='vendor-dropdown',
            options=[{'label': v, 'value': v} for v in leadtime_monthly_vendor['VendorName'].unique()],
            value=leadtime_monthly_vendor['VendorName'].unique()[0],
            clearable=False,
            style={'width': '50%'}
        ),
        dcc.Graph(id='leadtime-trend-line-chart')
    ], style={'backgroundColor': 'white', 'padding': '10px', 'margin': '10px'}),

    # EOQ Summary
    html.Div([
        html.H3("EOQ Summary"),
        html.Button("Export EOQ as CSV", id='download-eoq-btn', n_clicks=0, style={
            'marginBottom': '10px', 'backgroundColor': '#0074D9', 'color': 'white', 'padding': '10px',
            'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'
        }),
        dcc.Download(id="download-eoq-data"),
        dash_table.DataTable(
            id='eoq-table',
            columns=[{'name': col, 'id': col} for col in annual_demand.columns],
            data=[],
            page_size=10,
            filter_action='native',
            sort_action='native',
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center'},
            style_header={'fontWeight': 'bold'}
        )
    ], style={'backgroundColor': 'white', 'padding': '10px', 'margin': '10px'}),

    # Demand Forecasting
    html.Div([
        html.H3("Demand Forecasting by Product Description"),
        dcc.Dropdown(
            id='forecast-description-dropdown',
            options=[
                {'label': desc, 'value': desc}
                for desc in sorted(daily_sales['Description'].unique())
            ],
            value=sorted(daily_sales['Description'].unique())[0],
            clearable=False,
            style={'width': '50%', 'marginBottom': '10px'}
        ),
        dcc.Graph(id='forecast-graph')
    ], style={'backgroundColor': 'white', 'padding': '10px', 'margin': '10px'}),
])

# ABC Chart Callback
@app.callback(Output('abc-bar-chart', 'figure'), Input('abc-visual-dropdown', 'value'))
def update_abc_chart(chart_type):
    df = brand_sales.copy()
    if chart_type == 'bar':
        top = df.groupby('ABC_Category').apply(lambda x: x.nlargest(10, 'SalesDollars')).reset_index(drop=True)
        fig = px.bar(top, x='Description', y='SalesDollars', color='ABC_Category',
                     title="Top Brands by Sales in ABC Categories", barmode='group')
        fig.update_layout(xaxis_tickangle=-45, height=600, template='plotly_white', margin=dict(b=150))
    elif chart_type == 'pie':
        pie_df = df.groupby('ABC_Category')['SalesDollars'].sum().reset_index()
        fig = px.pie(pie_df, values='SalesDollars', names='ABC_Category', title="Sales Distribution by ABC Category")
    elif chart_type == 'box':
        fig = px.box(df, x='ABC_Category', y='SalesDollars', points='all',
                     title="Box Plot of Sales Dollars by ABC Category")
    else:
        fig = go.Figure()
    return fig

# Table Updates
@app.callback(Output('leadtime-table-vendor', 'data'), Input('leadtime-table-vendor', 'id'))
def update_leadtime_vendor(_): return leadtime_stats_vendor.to_dict('records')

@app.callback(Output('leadtime-table-brand', 'data'), Input('leadtime-table-brand', 'id'))
def update_leadtime_brand(_): return leadtime_stats_brand.to_dict('records')

@app.callback(Output('eoq-table', 'data'), Input('eoq-table', 'id'))
def update_eoq_table(_): return annual_demand.to_dict('records')

# Lead Time Trend
@app.callback(Output('leadtime-trend-line-chart', 'figure'), Input('vendor-dropdown', 'value'))
def update_leadtime_trend(selected_vendor):
    df = leadtime_monthly_vendor[leadtime_monthly_vendor['VendorName'] == selected_vendor]
    fig = px.line(df, x='YearMonth', y='LeadTimeDays', title=f"Lead Time Trend for Vendor: {selected_vendor}",
                  labels={'YearMonth': 'Month', 'LeadTimeDays': 'Lead Time (Days)'}, markers=True)
    fig.update_layout(template='plotly_white', height=500)
    return fig

# CSV Export
@app.callback(Output("download-vendor-data", "data"), Input("download-vendor-btn", "n_clicks"), prevent_initial_call=True)
def export_vendor_csv(n): return dcc.send_data_frame(leadtime_stats_vendor.to_csv, "leadtime_vendor.csv")

@app.callback(Output("download-brand-data", "data"), Input("download-brand-btn", "n_clicks"), prevent_initial_call=True)
def export_brand_csv(n): return dcc.send_data_frame(leadtime_stats_brand.to_csv, "leadtime_brand.csv")

@app.callback(Output("download-eoq-data", "data"), Input("download-eoq-btn", "n_clicks"), prevent_initial_call=True)
def export_eoq_csv(n): return dcc.send_data_frame(annual_demand.to_csv, "eoq_summary.csv")

# Forecast Graph
@app.callback(Output('forecast-graph', 'figure'), Input('forecast-description-dropdown', 'value'))
def update_forecast_graph(selected_description):
    df_desc = daily_sales[daily_sales['Description'] == selected_description].copy()
    if df_desc.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No sales data for description: {selected_description}")
        return fig
    forecast_df = moving_average_forecast(df_desc, window=7, forecast_days=forecast_horizon)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_desc['SalesDate'], y=df_desc['SalesQuantity'], mode='lines+markers', name='Historical Sales'))
    fig.add_trace(go.Scatter(x=forecast_df['SalesDate'], y=forecast_df['ForecastQuantity'], mode='lines+markers',
                             name='Forecasted Sales', line=dict(dash='dash', color='red')))
    fig.update_layout(
        title=f"Demand Forecast for '{selected_description}' (Next {forecast_horizon} days)",
        xaxis_title="Date",
        yaxis_title="Sales Quantity",
        template='plotly_white',
        height=500
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)
