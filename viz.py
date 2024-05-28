# viz.py

import streamlit as st
import pandas as pd
from plotly.subplots import make_subplots
import plotly.express as px
from millify import prettify

def meanChart(total_df, sgg_nm):
    st.markdown('## 가구별 평균 가격 추세 \n')

    filtered_df = total_df[total_df['SGG_NM'] == sgg_nm]
    filtered_df = filtered_df[filtered_df['DEAL_YMD'].between('2024-01-01', '2024-04-06')]
    result = filtered_df.groupby(['DEAL_YMD', 'HOUSE_TYPE'])['OBJ_AMT'].agg('mean').reset_index()

    df1 = result[result['HOUSE_TYPE'] == '아파트']
    df2 = result[result['HOUSE_TYPE'] == '단독다가구']
    df3 = result[result['HOUSE_TYPE'] == '오피스텔']
    df4 = result[result['HOUSE_TYPE'] == '연립다세대']

    fig = make_subplots(rows=2, cols=2, shared_xaxes=True,
                        subplot_titles=('아파트', '단독다가구', '오피스텔', '연립다세대'),
                        horizontal_spacing=0.15)
    fig.add_trace(px.line(df1, x='DEAL_YMD', y='OBJ_AMT', title='아파트 실거래가 평균', markers=True).data[0], row=1, col=1)
    fig.add_trace(px.line(df2, x='DEAL_YMD', y='OBJ_AMT', title='단독다가구 실거래가 평균', markers=True).data[0], row=1, col=2)
    fig.add_trace(px.line(df3, x='DEAL_YMD', y='OBJ_AMT', title='오피스텔 실거래가 평균', markers=True).data[0], row=2, col=1)
    fig.add_trace(px.line(df4, x='DEAL_YMD', y='OBJ_AMT', title='연립다세대 실거래가 평균', markers=True).data[0], row=2, col=2)

    fig.update_yaxes(tickformat='.0f', title_text='물건가격(원)', range=[result['OBJ_AMT'].min()-(result['OBJ_AMT'].min()/10), result['OBJ_AMT'].max()+(result['OBJ_AMT'].max()/10)])

    fig.update_layout(title = '가구별 평균값 추세 그래프', width=800, height=600, showlegend=True, template='plotly_white')
    
    st.plotly_chart(fig)


def deal_count(total_df, sgg_nm):
    st.markdown('## 가구별, 일별 거래 건수 추세\n')

    filtered_df = total_df[total_df['SGG_NM'] == sgg_nm]
    filtered_df = filtered_df[filtered_df['DEAL_YMD'].between('2024-01-01', '2024-04-06')]
    result = filtered_df.groupby(['DEAL_YMD', 'HOUSE_TYPE'])['OBJ_AMT'].count().reset_index().rename(columns = {'OBJ_AMT':'거래건수'})

    df1 = result[result['HOUSE_TYPE'] == '아파트']
    df2 = result[result['HOUSE_TYPE'] == '단독다가구']
    df3 = result[result['HOUSE_TYPE'] == '오피스텔']
    df4 = result[result['HOUSE_TYPE'] == '연립다세대']

    fig = make_subplots(rows=2, cols=2, shared_xaxes=True,
                        subplot_titles=('아파트', '단독다가구', '오피스텔', '연립다세대'),
                        horizontal_spacing=0.15)
    fig.add_trace(px.line(df1, x='DEAL_YMD', y='거래건수', title='아파트 거래건수(건)', markers=True).data[0], row=1, col=1)
    fig.add_trace(px.line(df2, x='DEAL_YMD', y='거래건수', title='단독다가구 거래건수(건)', markers=True).data[0], row=1, col=2)
    fig.add_trace(px.line(df3, x='DEAL_YMD', y='거래건수', title='오피스텔 거래건수(건)', markers=True).data[0], row=2, col=1)
    fig.add_trace(px.line(df4, x='DEAL_YMD', y='거래건수', title='연립다세대 거래건수(건)', markers=True).data[0], row=2, col=2)
    
    fig.update_yaxes(tickformat='.0f', title_text='건수', range=[-(result['거래건수'].max()/10), result['거래건수'].max() + (result['거래건수'].max()/20)])
    fig.update_layout(title = '가구별, 일별 거래건수 추세 그래프', width=800, height=600, showlegend=True, template='plotly_white')
    st.plotly_chart(fig)


def month_deal_count(total_df, sgg_nm):
    st.markdown('## 가구별, 월별 거래 건수 추세\n')

    filtered_df = total_df[total_df['SGG_NM'] == sgg_nm]
    filtered_df = filtered_df[filtered_df['DEAL_YMD'].between('2024-01-01', '2024-04-06')]

    filtered_df['DEAL_YMD'] = filtered_df['DEAL_YMD'].astype(str)
    filtered_df['DEAL_YM'] = filtered_df['DEAL_YMD'].str[:7] # 연도와 월만 추출
    result = filtered_df.groupby(['DEAL_YM', 'HOUSE_TYPE'])['OBJ_AMT'].count().reset_index().rename(columns = {'OBJ_AMT':'거래건수'})

    df1 = result[result['HOUSE_TYPE'] == '아파트']
    df2 = result[result['HOUSE_TYPE'] == '단독다가구']
    df3 = result[result['HOUSE_TYPE'] == '오피스텔']
    df4 = result[result['HOUSE_TYPE'] == '연립다세대']

    fig = make_subplots(rows=2, cols=2, shared_xaxes=True,
                        subplot_titles=('아파트', '단독다가구', '오피스텔', '연립다세대'),
                        horizontal_spacing=0.15)
    
    fig.add_trace(px.line(df1, x='DEAL_YM', y='거래건수', title='아파트 거래건수(건)', markers=True).data[0], row=1, col=1)
    fig.add_trace(px.line(df2, x='DEAL_YM', y='거래건수', title='단독다가구 거래건수(건)', markers=True).data[0], row=1, col=2)
    fig.add_trace(px.line(df3, x='DEAL_YM', y='거래건수', title='오피스텔 거래건수(건)', markers=True).data[0], row=2, col=1)
    fig.add_trace(px.line(df4, x='DEAL_YM', y='거래건수', title='연립다세대 거래건수(건)', markers=True).data[0], row=2, col=2)
    
    fig.update_yaxes(tickformat='.0f', title_text='건수', range=[-(result['거래건수'].max()/10), result['거래건수'].max() + (result['거래건수'].max()/20)])
    fig.update_layout(title = '가구별, 월별 거래건수 추세 그래프', width=800, height=600, showlegend=True, template='plotly_white')
    
    st.plotly_chart(fig)




def barChart(total_df):
    st.markdown('## 지역별 평균가격 막대 그래프\n')

    month_selected = st.selectbox("월을 선택하시오.", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    house_selected = st.selectbox("가구 유형을 선택하시오.", total_df['HOUSE_TYPE'].unique())

    total_df['month'] = total_df['DEAL_YMD'].dt.month
    result = total_df[(total_df['month'] == month_selected) & (total_df['HOUSE_TYPE'] == house_selected)]

    bar_df = result.groupby('SGG_NM')['OBJ_AMT'].agg('mean').reset_index()

    df_sorted = bar_df.sort_values('OBJ_AMT', ascending=False)

    fig = px.bar(df_sorted, x = 'SGG_NM', y = 'OBJ_AMT')

    fig.update_yaxes(tickformat='.0f', title_text = '물건 가격(만원)', range=[0, df_sorted['OBJ_AMT'].max()])
    fig.update_layout(title='Bar Chart - 오름차순', xaxis_title='지역구명', yaxis_title='평균가격(만원)')

    st.plotly_chart(fig)


def showViz(total_df):
    total_df['DEAL_YMD'] = pd.to_datetime(total_df['DEAL_YMD'], format='%Y-%m-%d')
    # total_df.dropna(subset=['SGG_NM'], inplace=True)
    sgg_nm = st.sidebar.selectbox('자치구명', total_df['SGG_NM'].unique())
    selected = st.sidebar.radio('차트메뉴', ['가구당 평균 가격 추세', '가구별, 일별 거래 건수 추세', '가구별, 월별 거래 건수 추세', '지역별 평균 가격 막대 그래프'])

    if selected == '가구당 평균 가격 추세':
        meanChart(total_df, sgg_nm)
    elif selected == '가구별, 일별 거래 건수 추세':
        deal_count(total_df, sgg_nm)
    elif selected == '가구별, 월별 거래 건수 추세':
        month_deal_count(total_df, sgg_nm)
    elif selected == '지역별 평균 가격 막대 그래프':
        barChart(total_df)
    else:
        st.warning("Error")

















