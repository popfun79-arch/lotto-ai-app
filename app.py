import os
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title='Lotto AI App',
    page_icon='🎯',
    layout='wide',
)

st.title('Lotto AI 분석 & 추천 앱')
st.write('Streamlit을 사용하여 분석 결과와 추천 번호를 웹에서 빠르게 확인합니다.')

base_path = os.path.abspath(os.path.dirname(__file__))

lotto_data_path = os.path.join(base_path, 'data', 'lotto_data.csv')
top10_path = os.path.join(base_path, 'results', 'top_10_recommended_combinations.csv')
weighted_path = os.path.join(base_path, 'results', 'weighted_combinations.csv')
prob_path = os.path.join(base_path, 'results', 'per_number_probabilities.csv')

@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

lotto_data = load_csv(lotto_data_path)
top10_df = load_csv(top10_path)
weighted_df = load_csv(weighted_path)
prob_df = load_csv(prob_path)

status_messages = []
if lotto_data is None:
    status_messages.append('`data/lotto_data.csv` 파일을 찾을 수 없습니다.')
if top10_df is None:
    status_messages.append('`results/top_10_recommended_combinations.csv` 파일을 찾을 수 없습니다.')
if weighted_df is None:
    status_messages.append('`results/weighted_combinations.csv` 파일을 찾을 수 없습니다.')
if prob_df is None:
    status_messages.append('`results/per_number_probabilities.csv` 파일을 찾을 수 없습니다.')

if lotto_data is None and top10_df is None and weighted_df is None and prob_df is None:
    st.error('필요한 데이터 파일이 없습니다. `data/lotto_data.csv` 및 `results/` 폴더를 확인하세요.')
    st.write('\n'.join(status_messages))
    st.stop()

st.sidebar.header('데이터 상태')
for msg in status_messages:
    st.sidebar.warning(msg)

if lotto_data is not None:
    st.sidebar.metric('데이터 행 수', len(lotto_data))
if top10_df is not None:
    st.sidebar.metric('Top 10 조합 수', len(top10_df))
if weighted_df is not None:
    st.sidebar.metric('가중 조합 수', len(weighted_df))
if prob_df is not None:
    st.sidebar.metric('확률 분석 번호 수', len(prob_df))

with st.expander('데이터 파일 경로 확인'):
    st.write(f'`data/lotto_data.csv`: {os.path.exists(lotto_data_path)}')
    st.write(f'`results/top_10_recommended_combinations.csv`: {os.path.exists(top10_path)}')
    st.write(f'`results/weighted_combinations.csv`: {os.path.exists(weighted_path)}')
    st.write(f'`results/per_number_probabilities.csv`: {os.path.exists(prob_path)}')

col1, col2 = st.columns(2)
if lotto_data is not None:
    with col1:
        latest_draw = lotto_data.sort_values('draw_num', ascending=False).iloc[0]
        st.metric('Latest Draw', int(latest_draw['draw_num']))
        st.write('### 최신 당첨 번호')
        st.write(
            f"{int(latest_draw['num1'])}, {int(latest_draw['num2'])}, {int(latest_draw['num3'])}, {int(latest_draw['num4'])}, {int(latest_draw['num5'])}, {int(latest_draw['num6'])} (보너스 {int(latest_draw['bonus'])})"
        )
        st.write('### 최근 데이터 요약')
        st.write(f'- 총 회차: {len(lotto_data)}')
        st.write(f'- 날짜 범위: {lotto_data.iloc[0].date} ~ {lotto_data.iloc[-1].date}' if 'date' in lotto_data.columns else '- 날짜 정보 없음')

if prob_df is not None:
    with col2:
        st.write('### 차기 출현 확률 Top 10')
        top_probs = prob_df.sort_values('prob_next', ascending=False).head(10)
        fig = px.bar(
            top_probs,
            x='number',
            y='prob_next',
            labels={'number': '번호', 'prob_next': '출현 확률'},
            title='차기 번호 확률 상위 10',
            text=top_probs['prob_next'].apply(lambda x: f'{x:.2%}')
        )
        fig.update_layout(yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)
        st.write('#### 확률 데이터 미리보기')
        st.dataframe(top_probs)

st.markdown('---')

st.write('## 추천 조합')
combo_tab, weight_tab = st.tabs(['Top 10 추천 조합', '가중 조합'])
with combo_tab:
    if top10_df is not None:
        st.write('### Top 10 추천 조합')
        numbered = top10_df.copy()
        numbered.index = numbered.index + 1
        numbered.index.name = '순위'
        st.table(numbered)
        st.download_button(
            label='Top 10 추천 조합 CSV 다운로드',
            data=numbered.to_csv(index=False, encoding='utf-8-sig'),
            file_name='top_10_recommended_combinations.csv',
            mime='text/csv',
        )
    else:
        st.warning('`results/top_10_recommended_combinations.csv` 파일을 찾을 수 없습니다.')
with weight_tab:
    if weighted_df is not None:
        st.write('### 가중 조합 샘플')
        st.dataframe(weighted_df.head(20))
        st.info(f'전체 가중 조합 수: {len(weighted_df)}')
        st.download_button(
            label='가중 조합 CSV 다운로드',
            data=weighted_df.to_csv(index=False, encoding='utf-8-sig'),
            file_name='weighted_combinations.csv',
            mime='text/csv',
        )
    else:
        st.warning('`results/weighted_combinations.csv` 파일을 찾을 수 없습니다.')

st.markdown('---')

if lotto_data is not None:
    st.write('## 데이터 탐색')
    st.write('### 최근 로또 데이터')
    st.dataframe(lotto_data.tail(10))
    st.write('### 번호 출현 빈도')
    freq = lotto_data[['num1','num2','num3','num4','num5','num6']].apply(pd.value_counts).sum(axis=1).sort_values(ascending=False).reset_index()
    freq.columns = ['number', 'count']
    fig2 = px.bar(
        freq,
        x='number',
        y='count',
        labels={'number': '번호', 'count': '출현 횟수'},
        title='번호 출현 빈도'
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.write('### 번호 출현 누적 순위')
    freq['rank'] = freq['count'].rank(method='min', ascending=False).astype(int)
    st.dataframe(freq)

st.sidebar.header('앱 정보')
st.sidebar.write('이 앱은 분석 결과 CSV 파일을 기반으로 동작합니다.')
if prob_df is not None:
    st.sidebar.write(f'- 확률 분석 번호: {len(prob_df)}개')
if top10_df is not None:
    st.sidebar.write(f'- Top 10 추천: {len(top10_df)}개')
if weighted_df is not None:
    st.sidebar.write(f'- 가중 조합: {len(weighted_df)}개')

st.write('---')
st.write('### 실행 방법')
st.code('streamlit run app.py', language='bash')
