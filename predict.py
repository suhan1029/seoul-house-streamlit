import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
from prophet import Prophet
import pandas as pd
import matplotlib.font_manager as fm
import numpy as np
from prophet.plot import plot_plotly

def predict_4(total_df):
    path = 'C:/Windows/Fonts/H2MJRE.TTF'
    fontprop = fm.FontProperties(fname=path, size=12)

    total_df['DEAL_YMD'] = pd.to_datetime(total_df['DEAL_YMD'], format='%Y-%m-%d')
    types = list(total_df['HOUSE_TYPE'].unique())
    periods=28

    fig, ax = plt.subplots(figsize=(10, 6), sharex=True, ncols=2, nrows=2)
    for i in range(0, len(types)):
        model = Prophet()

        total_df2 = total_df.loc[total_df['HOUSE_TYPE'] == types[i], ['DEAL_YMD', 'OBJ_AMT']]
        summary_df = total_df2.groupby('DEAL_YMD')['OBJ_AMT'].agg('mean').reset_index()
        summary_df = summary_df.rename(columns={'DEAL_YMD':'ds', 'OBJ_AMT':'y'})

        model.fit(summary_df)

        future1 = model.make_future_dataframe(periods=periods)

        forecast1 = model.predict(future1)

        x = i // 2
        y = i % 2

        fig = model.plot(forecast1, uncertainty=True, ax=ax[x, y])
        ax[x, y].set_title(f'서울시 {types[i]} 평균 가격 예측 시나리오 {periods}일간', fontproperties=fontprop)
        ax[x, y].set_xlabel(f'날짜', fontproperties=fontprop)
        ax[x, y].set_ylabel(f'평균가격(만원)', fontproperties=fontprop)
        for tick in ax[x, y].get_xticklabels():
            tick.set_rotation(30)

    plt.tight_layout()
    st.pyplot(plt)

def predict_25(total_df):
    # 한글폰트 설정
    path = 'C:/Windows/Fonts/H2MJRE.TTF'
    fontprop = fm.FontProperties(fname=path, size=12)

    total_df['DEAL_YMD'] = pd.to_datetime(total_df['DEAL_YMD'], format='%Y-%m-%d')

    total_df = total_df[total_df['HOUSE_TYPE'] == '아파트']

    sgg_nms = list(total_df['SGG_NM'].unique())
    print(sgg_nms)

    sgg_nms = [x for x in sgg_nms if x is not np.nan]
    print(sgg_nms)

    periods=28

    fig, ax = plt.subplots(figsize=(20, 10), sharex=True, sharey=False, ncols=5, nrows=5)

    loop = 0

    for sgg_nm in sgg_nms:
        model = Prophet()
        total_df2 = total_df.loc[total_df['SGG_NM'] == sgg_nm, ['DEAL_YMD', 'OBJ_AMT']]

        summary_df = total_df2.groupby('DEAL_YMD')['OBJ_AMT'].agg('mean').reset_index()
        summary_df = summary_df.rename(columns={'DEAL_YMD':'ds', 'OBJ_AMT':'y'})

        print(sgg_nm)

        model.fit(summary_df)

        future = model.make_future_dataframe(periods=periods)

        forcast = model.predict(future)

        x = loop // 5
        y = loop % 5
        loop += 1

        fig = model.plot(forcast, uncertainty=True, ax=ax[x, y])
        ax[x, y].set_title(f'서울시 {sgg_nm} 평균 가격 예측 시나리오 {periods}일간', fontproperties=fontprop)
        ax[x, y].set_xlabel(f'날짜', fontproperties=fontprop)
        ax[x, y].set_ylabel(f'평균가격(만원)', fontproperties=fontprop)
        for tick in ax[x, y].get_xticklabels():
            tick.set_rotation(30)

    plt.tight_layout()
    st.pyplot(plt)

def home():
    st.markdown("### 부동산 예측 개요\n"
                "- 가구별 평균 가격 예측\n"
                "- 지자체별 평균 가격 예측\n")

def reportMain(total_df):
    sgg_nm = st.sidebar.selectbox('자치구', total_df['SGG_NM'].unique())
    periods = int(st.number_input('향후 예측 기간을 지정하세요(1~30일)',
                  min_value=1, max_value=30, step=1))
    
    model = Prophet()

    total_df2 = total_df.loc[total_df['SGG_NM'] == sgg_nm, ['DEAL_YMD', 'OBJ_AMT']]

    summary_df = total_df2.groupby('DEAL_YMD')['OBJ_AMT'].agg('mean').reset_index()
    summary_df = summary_df.rename(columns = {'DEAL_YMD':'ds', 'OBJ_AMT':'y'})

    model.fit(summary_df)

    future = model.make_future_dataframe(periods=periods)

    forecast = model.predict(future)

    csv = forecast.to_csv(index=False).encode('utf-8')

    st.sidebar.download_button('결과 다운로드(CSV)', csv, f'{sgg_nm}_아파트 평균 예측 {periods}일간.csv', 'text/csv', key='download-csv')

    fig = plot_plotly(model, forecast)
    fig.update_layout(
        title=dict(text=f'{sgg_nm}_아파트 평균 예측 {periods}일간.csv', font=dict(size=20), yref='paper'),
        xaxis_title='날짜',
        yaxis_title='아파트 평균값(만원)',
        autosize=False,
        width=700,
        height=800
    )
    fig.update_yaxes(tickformat='000')
    st.plotly_chart(fig)

def predict_home(total_df):
    st.markdown("### 부동산 예측 \n"
                "부동산 예측 페이지 입니다."
                "여기에 포함하고 싶은 내용을 넣을 수 있습니다.")
    
    selected = option_menu(None, ['Home', '주거형태별', '자치구별', '보고서'],
                           icons=['house', 'bar-chart', 'file-spreadsheet', 'map'],
                           menu_icon='cast', default_index=0, orientation='horizontal',
                           styles={
                               'container':{
                                   'padding':'0!important',
                                   'background-color':'#808080'},
                               'icon':{
                                   'color':'orange',
                                   'font-size':'25px'},
                               'nav-link-selected':{
                                   'background-color':'green'}
                           })
    if selected == 'Home':
        home()
    elif selected == '주거형태별':
        predict_4(total_df)
    elif selected == '자치구별':
        predict_25(total_df)
    elif selected == '보고서':
        reportMain(total_df)
    else:
        st.warning('Wrong')