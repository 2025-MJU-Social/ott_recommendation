import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
import seaborn as sns
import warnings
import plotly.express as px
import plotly.graph_objects as go
import os
import matplotlib
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

# 저장 디렉토리 생성
os.makedirs('./EDA', exist_ok=True)

# 한글 폰트 경로 및 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
fontprop = FontProperties(fname=font_path)
font_name = font_manager.FontProperties(fname=font_path).get_name()
matplotlib.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False


def analyze_ott_service_experience(df_service_exp, save_path='./EDA/ott_service_usage.png', show_plot=False):
    """
    OTT 서비스 이용 경험 데이터를 분석하고 시각화하는 함수
    
    Args:
        df_service_exp (DataFrame): OTT 서비스 이용 경험 데이터를 담은 데이터프레임
        save_path (str): 저장할 이미지 파일 경로
        show_plot (bool): 그래프 표시 여부
    
    Returns:
        fig: Plotly 시각화 객체
    """
    # 1. 데이터에서 첫 번째 행을 헤더로 설정
    df = df_service_exp[1:].copy()
    df.columns = df_service_exp.iloc[0]
    
    # 2. 전체 행만 추출 (예: 전체 국민 기준)
    df_total = df[df['구분별(1)'] == '전체']
    
    # 3. 필요한 열만 선택하여 melt로 long-format 변환
    service_columns = df_total.columns[3:]  # '2023.1' ~ '2023.14'까지가 서비스 이용률
    df_melted = df_total.melt(id_vars=['구분별(1)', '구분별(2)'], 
                              value_vars=service_columns, 
                              var_name='서비스명', 
                              value_name='이용률')
    
    # 4. 데이터 타입 변환
    df_melted['이용률'] = pd.to_numeric(df_melted['이용률'], errors='coerce')
    
    # 5. 시각화
    fig = px.bar(df_melted, 
                 x='서비스명', 
                 y='이용률', 
                 title='2023년 전체 OTT 서비스별 이용 경험 비율',
                 labels={'이용률': '이용률 (%)'})
    
    # 6. 이미지 저장 - 대체 방법 사용
    try:
        print(f"Matplotlib로 이미지 저장: {save_path}")

        plt.figure(figsize=(12, 8))
        plt.bar(df_melted['서비스명'], df_melted['이용률'])
        plt.title('2023년 전체 OTT 서비스별 이용 경험 비율')
        plt.xlabel('서비스명')
        plt.ylabel('이용률 (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path)
        
        print(f"이미지 저장 성공: {save_path}")
    except Exception as e:
        print(f"Plotly로 이미지 저장 실패: {e}")
        
    
    # 7. 그래프 표시 (선택적)
    if show_plot:
        fig.show()
    
    return fig

def analyze_ott_usage_period(df_usage_period, save_path='./EDA/ott_usage_period.png', show_plot=False):
    """
    OTT 서비스 이용 기간 데이터를 분석하고 파이 차트로 시각화하는 함수
    
    Args:
        df_usage_period (DataFrame): OTT 서비스 이용 기간 데이터를 담은 데이터프레임
        save_path (str): 저장할 이미지 파일 경로
        show_plot (bool): 그래프 표시 여부
    
    Returns:
        fig: Plotly 시각화 객체
    """
    # 1. 원본 데이터프레임에서 첫 행을 컬럼으로 설정
    df = df_usage_period.copy()
    df.columns = df.iloc[0]  # 첫 번째 행을 열 이름으로
    df = df[1:]  # 데이터 본문만 남김
    
    # 2. '전체' 기준 행 추출
    df_total = df[df['구분별(1)'] == '전체']
    
    # 3. 시각화용 데이터프레임 생성
    # 기간별 컬럼만 추출
    period_columns = [
        '3개월 미만 (%)', '3개월-6개월 미만 (%)', '6개월-1년 미만 (%)',
        '1년-1년 6개월 미만 (%)', '1년 6개월-2년 미만 (%)', '2년 이상 (%)', '비이용 (%)'
    ]
    
    df_pie = pd.DataFrame({
        '이용기간': period_columns,
        '비율': [float(df_total[col].values[0]) for col in period_columns]
    })
    
    # 4. 파이 차트 시각화
    fig = px.pie(df_pie, 
                 names='이용기간',
                 values='비율',
                 title='2023년 전체 OTT 서비스 이용 기간 분포',
                 hole=0.3,  # 도넛 차트로 만들기 (선택 사항)
                 color_discrete_sequence=px.colors.qualitative.Pastel)  # 색상 설정 (선택 사항)
    
    fig.update_layout(
        legend_title='이용 기간',
        annotations=[dict(text='OTT 이용 기간', x=0.5, y=0.5, font_size=15, showarrow=False)]  # 도넛 차트 중앙 텍스트
    )
    
    # 5. 이미지 저장 - 대체 방법 사용
    try:
        print(f"Matplotlib로 이미지 저장: {save_path}")

        plt.figure(figsize=(10, 10))
        plt.pie(df_pie['비율'], labels=df_pie['이용기간'], autopct='%1.1f%%')
        plt.title('2023년 전체 OTT 서비스 이용 기간 분포')
        plt.tight_layout()
        plt.savefig(save_path)

        print(f"이미지 저장 성공: {save_path}")
    except Exception as e:
        print(f"Plotly로 이미지 저장 실패: {e}")
        
    # 6. 그래프 표시 (선택적)
    if show_plot:
        fig.show()
    
    return fig

def analyze_ott_continue_intent(df_continue_intent, save_path='./EDA/ott_continue_intent.png', show_plot=False):
    """
    OTT 서비스별 계속 이용 의향 데이터를 분석하고 시각화하는 함수
    
    Args:
        df_continue_intent (DataFrame): OTT 서비스 계속 이용 의향 데이터를 담은 데이터프레임
        save_path (str): 저장할 이미지 파일 경로
        show_plot (bool): 그래프 표시 여부
    
    Returns:
        fig: Plotly 시각화 객체
    """
    # 1. 헤더 행 적용
    df = df_continue_intent.copy()
    df.columns = df.iloc[0]  # 첫 번째 행을 열 이름으로 설정
    df = df[1:]  # 본문 데이터만 남김
    
    # 2. 전체 기준 행 필터링
    df_total = df[df['구분별(1)'] == '전체']
    
    # 3. 시각화용 서비스 컬럼 목록 정의
    service_columns = [
        '웨이브 (%)', '티빙 (%)', 'U+모바일 TV (%)', '왓챠 (%)', '카카오TV (%)',
        '유튜브 (%)', '넷플릭스 (%)', '아프리카 TV (%)', '디즈니플러스 (%)',
        '쿠팡플레이 (%)', '애플TV+ (%)', '기타 (%)'
    ]
    
    # 4. 새로운 DataFrame 구성
    df_bar = pd.DataFrame({
        '서비스명': service_columns,
        '계속이용의향비율': [float(df_total[col].values[0]) for col in service_columns]
    })
    
    # 5. 시각화 (Plotly)
    fig = px.bar(df_bar, 
                 x='서비스명', 
                 y='계속이용의향비율', 
                 title='OTT 서비스별 계속 이용 의향',
                 labels={'계속이용의향비율': '계속 이용 의향 (%)'})
    fig.update_layout(font=dict(family='Arial, sans-serif'))
    
    # 6. Matplotlib로 이미지 저장
    try:
        print(f"Matplotlib로 이미지 저장 시도: {save_path}")
        
        plt.figure(figsize=(12, 8))
        plt.bar(df_bar['서비스명'], df_bar['계속이용의향비율'])
        plt.title('OTT 서비스별 계속 이용 의향')
        plt.xlabel('서비스명')
        plt.ylabel('계속 이용 의향 (%)')
        plt.xticks(rotation=45, ha='right')  # 라벨 겹침 방지
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()  # 그래프 창 닫기 (중요)
        
        print(f"이미지 저장 성공: {save_path}")
    except Exception as e:
        print(f"이미지 저장 실패: {e}")
    
    # 7. 그래프 표시 (선택적)
    if show_plot:
        fig.show()
    
    return fig

def analyze_ott_usage_frequency(df_usage_freq, save_path='./EDA/ott_usage_frequency.png', show_plot=False):
    """
    OTT 서비스 이용 빈도 데이터를 분석하고 시각화하는 함수
    
    Args:
        df_usage_freq (DataFrame): OTT 서비스 이용 빈도 데이터를 담은 데이터프레임
        save_path (str): 저장할 이미지 파일 경로
        show_plot (bool): 그래프 표시 여부
    
    Returns:
        fig: Plotly 시각화 객체
    """
    # 1. 헤더 설정
    df = df_usage_freq.copy()
    df.columns = df.iloc[0]  # 첫 행을 컬럼명으로 설정
    df = df[1:]  # 실제 데이터만 추출
    
    # 2. 전체 기준 행 추출
    df_total = df[df['구분별(1)'] == '전체']
    
    # 3. 이용 빈도 관련 열 추출
    freq_columns = [
        '주 1일 미만 (%)', '주 1-2일 (%)', '주 3-4일 (%)',
        '주 5-6일 (%)', '주 7일 (%)', 'OTT 비이용 (%)'
    ]
    
    # 4. 시각화용 데이터프레임 생성
    df_bar = pd.DataFrame({
        '이용빈도': freq_columns,
        '비율': [float(df_total[col].values[0]) for col in freq_columns]
    })
    
    # 5. Plotly 시각화
    fig = px.bar(df_bar,
                 x='이용빈도',
                 y='비율',
                 title='2023년 전체 OTT 서비스 이용 빈도 분포',
                 labels={'비율': '비율 (%)'},
                 color='비율',  # 색상 그라데이션 추가
                 color_continuous_scale='Viridis')  # 색상 스키마 설정
    
    fig.update_layout(
        font=dict(family='Arial, sans-serif'),
        xaxis_title='이용 빈도',
        yaxis_title='비율 (%)',
        coloraxis_showscale=False  # 색상 스케일 바 숨기기
    )
    
    # 6. Matplotlib로 이미지 저장
    try:
        print(f"Matplotlib로 이미지 저장 시도: {save_path}")
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(df_bar['이용빈도'], df_bar['비율'], 
                       color=plt.cm.viridis(np.linspace(0, 1, len(freq_columns))))  # 컬러 그라데이션
        
        # 바 위에 값 표시
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.title('2023년 전체 OTT 서비스 이용 빈도 분포')
        plt.xlabel('이용 빈도')
        plt.ylabel('비율 (%)')
        plt.xticks(rotation=45, ha='right')  # 라벨 겹침 방지
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()  # 그래프 창 닫기 (중요)
        
        print(f"이미지 저장 성공: {save_path}")
    except Exception as e:
        print(f"이미지 저장 실패: {e}")
    
    # 7. 그래프 표시 (선택적)
    if show_plot:
        fig.show()
    
    return fig

def process_multiple_comma_separated_columns(df, columns_to_process):
    """
    데이터프레임에서 콤마로 구분된 값이 있는 여러 컬럼을 처리하고 분석 결과 반환.
    
    각 값을 분리하여 리스트로 변환하고, explode로 확장된 데이터프레임과 값 빈도를 계산.
    
    Args:
        df (DataFrame): 처리할 데이터프레임
        columns_to_process (list): 콤마로 구분된 값이 있는 컬럼명 리스트
    
    Returns:
        dict: 각 컬럼별 처리 결과(파싱된 시리즈, 확장된 데이터프레임, 값 빈도 등)를 담은 딕셔너리
    """
    results = {}
    
    for column in columns_to_process:
        if column not in df.columns:
            print(f"경고: {column} 컬럼이 데이터프레임에 존재하지 않습니다.")
            continue
            
        # 해당 컬럼의 데이터가 문자열인지 확인하고 NaN 처리
        df[column] = df[column].fillna('')
        
        # 문자열이 아닌 값은 문자열로 변환
        df[column] = df[column].astype(str)
        
        # 쌍따옴표 제거 및 콤마+공백으로 분리
        parsed_series = df[column].apply(lambda x: 
                                         [item.strip() for item in x.strip('"').split(', ')] 
                                         if x.strip() else [])
        
        # 원본 데이터프레임에 파싱된 리스트 저장
        df_with_lists = df.copy()
        df_with_lists[column] = parsed_series
        
        # 분석을 위해 explode 사용하여 확장
        df_exploded = df_with_lists.explode(column)
        
        # 빈 값 제거
        df_exploded = df_exploded[df_exploded[column] != '']
        
        # 결과 저장
        results[column] = {
            'parsed_series': parsed_series,
            'df_with_lists': df_with_lists,
            'df_exploded': df_exploded,
            'value_counts': df_exploded[column].value_counts()
        }
    
    return results

# 통합 분석 함수
def analyze_drama_data(df, results, save_to_file=True, file_path='./EDA/analyze_drama_data.txt'):
    """
    드라마/영화 데이터 통합 분석 및 결과 저장
    먼저 title 중복을 처리한 후 분석을 진행합니다.
    참고: 중복 제거는 함수 내부에서만 사용되며, 원본 데이터는 변경되지 않습니다.
    
    Args:
        df (DataFrame): 원본 데이터프레임
        results (dict): process_multiple_comma_separated_columns 함수의 결과
        save_to_file (bool): 결과를 파일로 저장할지 여부
        file_path (str): 저장할 파일 경로
    
    Returns:
        list: 분석 결과 텍스트 라인 리스트
    """
    # 출력 내용을 저장할 리스트
    output_lines = []
    
    # 함수 내부에서만 사용할 데이터프레임 복사본 생성
    df_analysis = df.copy()
    results_analysis = {}
    for column, result in results.items():
        results_analysis[column] = {
            'df_exploded': result['df_exploded'].copy(),
            'value_counts': result['value_counts'].copy()
        }
    
    # 0. 중복 title 처리
    output_lines.append("=== 중복 Title 처리 ===")
    title_counts = df_analysis['title'].value_counts()
    duplicated_titles = title_counts[title_counts > 1].index.tolist()
    
    if duplicated_titles:
        output_lines.append(f"중복된 제목 수: {len(duplicated_titles)}")
        output_lines.append(f"중복된 제목 목록: {duplicated_titles}")
        
        # 중복 처리 방법: 각 중복 그룹 내에서 평점이 가장 높은 행만 유지
        output_lines.append("\n중복 처리 방법: 각 중복 그룹 내에서 평점이 가장 높은 행만 유지")
        
        # 중복 처리 전 행 수
        output_lines.append(f"중복 처리 전 행 수: {len(df_analysis)}")
        
        # 중복 처리: 평점이 높은 항목만 남기기
        df_no_duplicates = df_analysis.loc[df_analysis.groupby('title')['score'].idxmax()]
        
        # 중복 처리 후 행 수
        output_lines.append(f"중복 처리 후 행 수: {len(df_no_duplicates)}")
        
        # 제거된 행 수
        output_lines.append(f"제거된 행 수: {len(df_analysis) - len(df_no_duplicates)}")
    else:
        output_lines.append("중복된 제목이 없습니다.")
        df_no_duplicates = df_analysis.copy()
    
    # 중복이 처리된 데이터프레임으로 결과 객체 업데이트
    for column in results_analysis:
        if column in ['genre_detail', 'cast', 'platform']:
            exploded_df = results_analysis[column]['df_exploded']
            filtered_df = exploded_df[exploded_df['title'].isin(df_no_duplicates['title'])]
            results_analysis[column]['df_exploded'] = filtered_df
    
    # 이제 중복이 제거된 데이터프레임으로 분석 진행
    df_analysis = df_no_duplicates
    
    # 1. 기본 통계
    output_lines.append("\n=== 기본 통계 ===")
    output_lines.append(f"총 작품 수: {len(df_analysis)}")
    output_lines.append(f"평균 평점: {df_analysis['score'].mean():.2f}")
    output_lines.append(f"장르 분포: {df_analysis['genre'].value_counts().to_dict()}")
    
    # 2. 연도별 통계
    year_stats = df_analysis.groupby('year').agg({
        'score': ['mean', 'count']
    })
    output_lines.append("\n=== 연도별 통계 ===")
    output_lines.append(str(year_stats))
    
    # 3. 각 컬럼별 분석 결과
    for column, result in results_analysis.items():
        output_lines.append(f"\n=== {column} 분석 결과 ===")
        
        # exploded 데이터프레임으로 다시 계산
        exploded_df = result['df_exploded']
        if len(exploded_df) > 0:
            value_counts = exploded_df[column].value_counts()
            output_lines.append(f"고유 {column} 요소 수: {len(value_counts)}")
            output_lines.append(f"상위 5개 {column} 요소:")
            output_lines.append(str(value_counts.head(5)))
        else:
            output_lines.append(f"중복 제거 후 {column} 데이터가 없습니다.")
    
    # 4. 장르 상세 요소와 평점 간의 관계
    if 'genre_detail' in results_analysis:
        genre_detail_df = results_analysis['genre_detail']['df_exploded']
        if len(genre_detail_df) > 0:
            genre_detail_scores = genre_detail_df.groupby('genre_detail')['score'].agg(['mean', 'count'])
            genre_detail_scores = genre_detail_scores[genre_detail_scores['count'] >= 2].sort_values('mean', ascending=False)
            
            output_lines.append("\n=== 장르 상세 요소별 평균 평점 (2개 이상 작품이 있는 경우) ===")
            if len(genre_detail_scores) > 0:
                output_lines.append(str(genre_detail_scores.head(10)))
            else:
                output_lines.append("2개 이상 작품이 있는 장르 상세 요소가 없습니다.")
    
    # 5. 배우와 평점 간의 관계
    if 'cast' in results_analysis:
        cast_df = results_analysis['cast']['df_exploded']
        if len(cast_df) > 0:
            actor_scores = cast_df.groupby('cast')['score'].agg(['mean', 'count'])
            actor_scores = actor_scores[actor_scores['count'] >= 2].sort_values('mean', ascending=False)
            
            output_lines.append("\n=== 배우별 평균 평점 (2개 이상 작품에 출연한 경우) ===")
            if len(actor_scores) > 0:
                output_lines.append(str(actor_scores.head(10)))
            else:
                output_lines.append("2개 이상 작품에 출연한 배우가 없습니다.")
    
    # 6. 플랫폼과 평점 간의 관계
    if 'platform' in results_analysis:
        platform_df = results_analysis['platform']['df_exploded']
        if len(platform_df) > 0:
            platform_scores = platform_df.groupby('platform')['score'].agg(['mean', 'count'])
            
            output_lines.append("\n=== 플랫폼별 평균 평점 ===")
            if len(platform_scores) > 0:
                output_lines.append(str(platform_scores.sort_values('mean', ascending=False)))
            else:
                output_lines.append("플랫폼 데이터가 없습니다.")
    
    # 결과 출력
    for line in output_lines:
        print(line)
    
    # 파일로 저장 (선택적)
    if save_to_file:
        try:
            # 디렉토리 생성
            import os
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 파일로 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                for line in output_lines:
                    f.write(line + "\n")
            print(f"\n분석 결과가 '{file_path}'에 저장되었습니다.")
        except Exception as e:
            print(f"\n파일 저장 중 오류가 발생했습니다: {e}")
    
    return output_lines  # 분석 결과 텍스트만 반환

def visualize_drama_data(df, results, save_path='./EDA/drama_analysis_comprehensive.png', show=False, fontprop=fontprop):
    """
    드라마/영화 데이터 통합 시각화
    
    Args:
        df (DataFrame): 원본 데이터프레임
        results (dict): process_multiple_comma_separated_columns 함수의 결과
        save_path (str): 저장할 이미지 파일 경로
        show (bool): 그래프 표시 여부
        fontprop (FontProperties, optional): 한글 폰트 속성
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 2, figure=fig)

    # 1. 평점 분포
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(df['score'], bins=10, kde=True, ax=ax1)
    ax1.set_title('평점 분포', fontproperties=fontprop)
    ax1.set_xlabel('평점', fontproperties=fontprop)
    ax1.set_ylabel('작품 수', fontproperties=fontprop)
    for label in ax1.get_xticklabels():
        label.set_fontproperties(fontprop)
    for label in ax1.get_yticklabels():
        label.set_fontproperties(fontprop)

    # 2. 연도별 작품 수와 평균 평점
    ax2 = fig.add_subplot(gs[0, 1])
    yearly_stats = df.groupby('year').agg({'title': 'count','score': 'mean'}).reset_index()
    ax2_twin = ax2.twinx()
    sns.barplot(x='year', y='title', data=yearly_stats, color='skyblue', ax=ax2)
    sns.lineplot(x='year', y='score', data=yearly_stats, color='red', marker='o', ax=ax2_twin)
    ax2.set_xlabel('연도', fontproperties=fontprop)
    ax2.set_ylabel('작품 수', fontproperties=fontprop)
    ax2_twin.set_ylabel('평균 평점', fontproperties=fontprop, fontsize=12, color='red')
    for label in ax2.get_xticklabels():
        label.set_fontproperties(fontprop)
    for label in ax2.get_yticklabels():
        label.set_fontproperties(fontprop)
    for label in ax2_twin.get_yticklabels():
        label.set_fontproperties(fontprop)
        label.set_color('red')

    # 3. 장르 상세 요소 Top 10
    if 'genre_detail' in results:
        ax3 = fig.add_subplot(gs[1, 0])
        genre_counts = results['genre_detail']['value_counts']
        top_genres = genre_counts.head(10)
        bars = sns.barplot(x=top_genres.values, y=top_genres.index, ax=ax3)
        ax3.set_title('장르 상세 요소 Top 10', fontproperties=fontprop)
        ax3.set_xlabel('작품 수', fontproperties=fontprop)
        ax3.set_ylabel('장르 상세 요소', fontproperties=fontprop)
        for text in ax3.get_yticklabels():
            text.set_fontproperties(fontprop)

    # 4. 출연 빈도가 높은 배우 Top 10
    if 'cast' in results:
        ax4 = fig.add_subplot(gs[1, 1])
        actor_counts = results['cast']['value_counts']
        top_actors = actor_counts.head(10)
        bars = sns.barplot(x=top_actors.values, y=top_actors.index, ax=ax4)
        ax4.set_title('출연 빈도 Top 10 배우', fontproperties=fontprop)
        ax4.set_xlabel('작품 수', fontproperties=fontprop)
        ax4.set_ylabel('배우', fontproperties=fontprop)
        for text in ax4.get_yticklabels():
            text.set_fontproperties(fontprop)

    # 5. 플랫폼 분포
    if 'platform' in results:
        ax5 = fig.add_subplot(gs[2, 0])
        platform_counts = results['platform']['value_counts']
        bars = sns.barplot(x=platform_counts.values, y=platform_counts.index, ax=ax5)
        ax5.set_title('플랫폼 분포', fontproperties=fontprop)
        ax5.set_xlabel('작품 수', fontproperties=fontprop)
        ax5.set_ylabel('플랫폼', fontproperties=fontprop)
        for text in ax5.get_yticklabels():
            text.set_fontproperties(fontprop)

    # 6. 장르별 평균 평점
    ax6 = fig.add_subplot(gs[2, 1])
    genre_avg_scores = df.groupby('genre')['score'].mean().sort_values(ascending=False)
    bars = sns.barplot(x=genre_avg_scores.values, y=genre_avg_scores.index, ax=ax6)
    ax6.set_title('장르별 평균 평점', fontproperties=fontprop)
    ax6.set_xlabel('평균 평점', fontproperties=fontprop)
    ax6.set_ylabel('장르', fontproperties=fontprop)
    for text in ax6.get_yticklabels():
        text.set_fontproperties(fontprop)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

# 7. 심화 분석: 장르와 장르 상세 요소의 관계 시각화
def visualize_genre_relationships(results, save_path='./EDA/genre_relationship_heatmap.png', show=False, fontprop=fontprop):
    """
    장르와 장르 상세 요소의 관계 시각화
    
    Args:
        results (dict): process_multiple_comma_separated_columns 함수의 결과
        save_path (str): 저장할 이미지 파일 경로
        show (bool): 그래프 표시 여부
        fontprop (FontProperties, optional): 한글 폰트 속성
    """
    if 'genre_detail' not in results:
        print("장르 상세 요소 데이터가 없습니다.")
        return
    
    genre_detail_df = results['genre_detail']['df_exploded']
    
    # 장르와 장르 상세 요소의 관계를 크로스탭으로 분석
    genre_detail_cross = pd.crosstab(genre_detail_df['genre'], genre_detail_df['genre_detail'])
    
    # 히트맵으로 시각화
    plt.figure(figsize=(14, 10))
    
    # 히트맵 생성
    ax = sns.heatmap(genre_detail_cross, cmap='YlGnBu', linewidths=0.5, annot=True, fmt='d')
    
    # 한글 폰트 적용
    ax.set_title('장르와 장르 상세 요소의 관계', fontproperties=fontprop, fontsize=16)
    ax.set_xlabel('장르 상세 요소', fontproperties=fontprop, fontsize=12)
    ax.set_ylabel('장르', fontproperties=fontprop, fontsize=12)
    
    # X축과 Y축 라벨에 폰트 적용
    for label in ax.get_xticklabels():
        label.set_fontproperties(fontprop)
        label.set_rotation(45)  # X축 라벨 회전
        label.set_ha('right')   # 오른쪽 정렬
    
    for label in ax.get_yticklabels():
        label.set_fontproperties(fontprop)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

# 8. 네트워크 분석 (배우와 작품 간의 관계)
def visualize_actor_network(results, save_path='./EDA/actor_work_network.png', show=False, fontprop=fontprop):
    """
    배우와 작품 간의 관계 네트워크 시각화
    
    Args:
        results (dict): process_multiple_comma_separated_columns 함수의 결과
        save_path (str): 저장할 이미지 파일 경로
        show (bool): 그래프 표시 여부
        fontprop (FontProperties, optional): 한글 폰트 속성
    """
    try:
        import networkx as nx
        
        if 'cast' not in results:
            print("출연자 데이터가 없습니다.")
            return
        
        # 출연자와 작품 데이터
        cast_df = results['cast']['df_exploded']
        
        # 그래프 생성
        G = nx.Graph()
        
        # 각 작품과 출연자를 노드로 추가하고 연결
        for _, row in cast_df.iterrows():
            title = row['title']
            actor = row['cast']
            
            # 노드 속성 설정
            if not G.has_node(title):
                G.add_node(title, type='work', score=row['score'])
            
            if not G.has_node(actor):
                G.add_node(actor, type='actor')
            
            # 엣지 추가
            G.add_edge(title, actor)
        
        # 그래프 레이아웃 계산
        pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)  # seed 추가로 결과 일관성 유지
        
        # 시각화
        plt.figure(figsize=(16, 12))
        
        # 작품 노드와 배우 노드 구분
        work_nodes = [node for node, attrs in G.nodes(data=True) if attrs.get('type') == 'work']
        actor_nodes = [node for node, attrs in G.nodes(data=True) if attrs.get('type') == 'actor']
        
        # 작품 노드 (평점에 따라 크기 변화)
        work_scores = [G.nodes[node]['score'] for node in work_nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=work_nodes, 
                              node_size=[score*10 for score in work_scores],
                              node_color='lightblue', alpha=0.8)
        
        # 배우 노드
        nx.draw_networkx_nodes(G, pos, nodelist=actor_nodes, 
                              node_size=100, node_color='lightgreen', alpha=0.6)
        
        # 엣지
        nx.draw_networkx_edges(G, pos, alpha=0.2)
        
        # 라벨 (상위 연결성을 가진 노드만 표시)
        degree = dict(nx.degree(G))
        top_nodes = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:20]
        labels = {node: node for node, _ in top_nodes}
        
        # 한글 폰트 적용 (networkx는 font_family 대신 font_properties 사용)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_family='Malgun Gothic')
        plt.title('출연자와 작품 간의 관계 네트워크', fontproperties=fontprop, fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        
        # 중심성 분석
        betweenness = nx.betweenness_centrality(G)
        top_central = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print("=== 네트워크 중심성 높은 상위 10개 노드 ===")
        for node, score in top_central:
            node_type = G.nodes[node].get('type', 'unknown')
            print(f"{node} ({node_type}): {score:.4f}")
        
        return G
    
    except ImportError:
        print("networkx 라이브러리가 설치되어 있지 않습니다. 'pip install networkx'로 설치해주세요.")
        return None

# 9. 장르 상세 요소에 따른 평점 비교
def visualize_genre_detail_scores(results, save_path='./EDA/genre_detail_scores.png', show=False, fontprop=fontprop):
    """
    장르 상세 요소별 평균 평점 시각화
    
    Args:
        results (dict): process_multiple_comma_separated_columns 함수의 결과
        save_path (str): 저장할 이미지 파일 경로
        show (bool): 그래프 표시 여부
        fontprop (FontProperties, optional): 한글 폰트 속성
    """
    if 'genre_detail' not in results:
        print("장르 상세 요소 데이터가 없습니다.")
        return
    
    genre_detail_df = results['genre_detail']['df_exploded']
    
    # 장르 상세 요소별 평균 평점 및 작품 수 계산
    genre_detail_scores = genre_detail_df.groupby('genre_detail').agg({
        'score': ['mean', 'count']
    })
    
    # 평균 평점 기준으로 정렬
    genre_detail_scores = genre_detail_scores.sort_values(('score', 'mean'), ascending=False)
    
    # 작품이 2개 이상인 장르 요소만 필터링
    filtered_scores = genre_detail_scores[genre_detail_scores[('score', 'count')] >= 2]
    
    # 상위 15개 요소만 선택
    top_scores = filtered_scores.head(15).reset_index()
    
    # 그래프 생성
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 바 그래프 (평균 평점)
    bars = sns.barplot(x='genre_detail', y=('score', 'mean'), data=top_scores, 
                      palette='viridis', ax=ax)
    
    # 한글 폰트 적용 (제목 및 축 라벨)
    ax.set_title('장르 상세 요소별 평균 평점 (작품 수 2개 이상)', fontproperties=fontprop, fontsize=14)
    ax.set_xlabel('장르 상세 요소', fontproperties=fontprop, fontsize=12)
    ax.set_ylabel('평균 평점', fontproperties=fontprop, fontsize=12)
    
    # X축 라벨 폰트 및 회전 설정
    labels = ax.get_xticklabels()
    for label in labels:
        label.set_fontproperties(fontprop)
        label.set_rotation(45)
        label.set_ha('right')
    
    # Y축 라벨 폰트 설정
    for label in ax.get_yticklabels():
        label.set_fontproperties(fontprop)
        
    # 바 위에 표시되는 텍스트에도 폰트 적용 (평점)
    for i, bar in enumerate(bars.patches):
        text = bars.text(bar.get_x() + bar.get_width()/2., 
                       bar.get_height() + 0.3,
                       f"{top_scores[('score', 'mean')].iloc[i]:.1f}",
                       ha='center', va='bottom', fontsize=10)
        text.set_fontproperties(fontprop)
    
    # 바 내부 텍스트에도 폰트 적용 (작품 수)
    for i, bar in enumerate(bars.patches):
        text = bars.text(bar.get_x() + bar.get_width()/2., 
                       bar.get_height() / 2,
                       f"n={int(top_scores[('score', 'count')].iloc[i])}",
                       ha='center', va='center', fontsize=9, color='white', weight='bold')
        text.set_fontproperties(fontprop)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def visualize_platform_genre_heatmap(train_data, save_path='./EDA/platform_genre_heatmap.png', fontprop=fontprop):
    """
    플랫폼별, 장르별 작품 수를 heatmap으로 시각화하여 저장
    Args:
        train_data (DataFrame): 원본 데이터프레임 (platform, genre 컬럼 필요)
        save_path (str): 저장할 이미지 파일 경로
        fontprop (FontProperties): 한글 폰트 속성
    """
    df = train_data.copy()
    df['platform'] = df['platform'].fillna('').astype(str)
    df['genre'] = df['genre'].fillna('').astype(str)
    df['platform'] = df['platform'].apply(lambda x: [item.strip() for item in x.strip('"').split(', ') if item.strip()])
    df['genre'] = df['genre'].apply(lambda x: [item.strip() for item in x.strip('"').split(', ') if item.strip()])
    df_exploded = df.explode('platform').explode('genre')
    df_exploded = df_exploded[(df_exploded['platform'] != '') & (df_exploded['genre'] != '')]
    cross = df_exploded.groupby(['platform', 'genre']).size().unstack(fill_value=0)
    plt.figure(figsize=(max(10, len(cross.index)*0.8), max(8, len(cross.columns)*0.5)))
    ax = sns.heatmap(cross, annot=True, fmt='d', cmap='YlGnBu', cbar=True)
    plt.title('플랫폼별 장르별 작품 수', fontproperties=fontprop, fontsize=16)
    plt.xlabel('장르', fontproperties=fontprop, fontsize=12)
    plt.ylabel('플랫폼', fontproperties=fontprop, fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontproperties=fontprop)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=fontprop)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'플랫폼별 장르별 작품 수 heatmap이 {save_path}에 저장되었습니다.')

def visualize_platform_genre_detail_heatmap(train_data, save_path='./EDA/platform_genre_detail_heatmap.png', fontprop=fontprop):
    """
    플랫폼별, 장르별 작품 수를 heatmap으로 시각화하여 저장
    Args:
        train_data (DataFrame): 원본 데이터프레임 (platform, genre 컬럼 필요)
        save_path (str): 저장할 이미지 파일 경로
        fontprop (FontProperties): 한글 폰트 속성
    """
    df = train_data.copy()
    df['platform'] = df['platform'].fillna('').astype(str)
    df['genre_detail'] = df['genre_detail'].fillna('').astype(str)
    df['platform'] = df['platform'].apply(lambda x: [item.strip() for item in x.strip('"').split(', ') if item.strip()])
    df['genre_detail'] = df['genre_detail'].apply(lambda x: [item.strip() for item in x.strip('"').split(', ') if item.strip()])
    df_exploded = df.explode('platform').explode('genre_detail')
    df_exploded = df_exploded[(df_exploded['platform'] != '') & (df_exploded['genre_detail'] != '')]
    cross = df_exploded.groupby(['platform', 'genre_detail']).size().unstack(fill_value=0)
    plt.figure(figsize=(max(10, len(cross.index)*0.8), max(8, len(cross.columns)*0.5)))
    ax = sns.heatmap(cross, annot=True, fmt='d', cmap='YlGnBu', cbar=True)
    plt.title('플랫폼별 장르 상세 요소별 작품 수', fontproperties=fontprop, fontsize=16)
    plt.xlabel('장르 상세 요소', fontproperties=fontprop, fontsize=12)
    plt.ylabel('플랫폼', fontproperties=fontprop, fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontproperties=fontprop)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=fontprop)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'플랫폼별 장르 상세 요소별 작품 수 heatmap이 {save_path}에 저장되었습니다.')

def visualize_platform_genre_detail_stacked_bar(train_data, save_path='./EDA/platform_genre_detail_stacked_bar.png', fontprop=fontprop, top_n=7):
    """
    플랫폼별로 장르 상세 요소별 작품 수를 누적 막대그래프로 시각화 (상위 N개만, 기타 없음)
    """
    df = train_data.copy()
    df['platform'] = df['platform'].fillna('').astype(str)
    df['genre_detail'] = df['genre_detail'].fillna('').astype(str)
    df['platform'] = df['platform'].apply(lambda x: [item.strip() for item in x.strip('"').split(', ') if item.strip()])
    df['genre_detail'] = df['genre_detail'].apply(lambda x: [item.strip() for item in x.strip('"').split(', ') if item.strip()])
    df_exploded = df.explode('platform').explode('genre_detail')
    df_exploded = df_exploded[(df_exploded['platform'] != '') & (df_exploded['genre_detail'] != '')]
    # 상위 N개 장르 상세 요소만 추출
    top_genres = df_exploded['genre_detail'].value_counts().head(top_n).index
    df_exploded = df_exploded[df_exploded['genre_detail'].isin(top_genres)]
    # 집계
    count_df = df_exploded.groupby(['platform', 'genre_detail']).size().unstack(fill_value=0)
    # 상위 N개 컬럼만 남기기
    count_df = count_df[top_genres]
    count_df = count_df.loc[count_df.sum(axis=1).sort_values(ascending=False).index]  # 플랫폼별 합계 기준 정렬
    plt.figure(figsize=(max(10, len(count_df.index)*1.2), 8))
    bottom = np.zeros(len(count_df))
    colors = plt.cm.tab20.colors
    for i, genre in enumerate(count_df.columns):
        plt.bar(count_df.index, count_df[genre], bottom=bottom, label=genre, color=colors[i % len(colors)])
        bottom += count_df[genre].values
    plt.title('플랫폼별 장르 상세 요소별 작품 수 (상위 {}개)'.format(top_n), fontproperties=fontprop, fontsize=16)
    plt.xlabel('플랫폼', fontproperties=fontprop, fontsize=12)
    plt.ylabel('작품 수', fontproperties=fontprop, fontsize=12)
    plt.xticks(rotation=45, ha='right', fontproperties=fontprop)
    plt.legend(title='장르 상세 요소', prop=fontprop, bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'플랫폼별 장르 상세 요소별 작품 수 stacked bar가 {save_path}에 저장되었습니다.')

if __name__ == "__main__":

    matplotlib.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False

    # 데이터 로드
    df_continue_intent = pd.read_csv('./data/ott/OTT_유료서비스_계속_이용_의향__서비스별_20250413203427.csv', encoding='cp949')
    df_usage_period = pd.read_csv('./data/ott/OTT_유료서비스_이용기간__전체_20250413203445.csv', encoding='cp949')
    df_usage_exp = pd.read_csv('./data/ott/OTT_이용_경험_여부_20250413203126.csv', encoding='cp949')
    df_service_exp = pd.read_csv('./data/ott/OTT_이용_경험_여부_서비스별_20250413203230.csv', encoding='cp949')
    df_usage_freq = pd.read_csv('./data/ott/OTT_이용_빈도_20250413203151.csv', encoding='cp949')
    train_data = pd.read_csv('./data/train_data.csv')

    # 저장 경로 명시
    service_exp_path = './EDA/ott_service_usage.png'
    usage_period_path = './EDA/ott_usage_period.png'
    continue_intent_path = './EDA/ott_continue_intent.png'
    usage_freq_path = './EDA/ott_usage_frequency.png'

    print("서비스별 이용 경험 데이터 분석 중...")
    fig1 = analyze_ott_service_experience(df_service_exp, save_path=service_exp_path, show_plot=False)
    
    print("이용 기간 데이터 분석 중...")
    fig2 = analyze_ott_usage_period(df_usage_period, save_path=usage_period_path, show_plot=False)
    
    print("계속 이용 의향 데이터 분석 중...")
    fig3 = analyze_ott_continue_intent(df_continue_intent, save_path=continue_intent_path, show_plot=False)

    print("이용 빈도 데이터 분석 중...")
    fig4 = analyze_ott_usage_frequency(df_usage_freq, save_path=usage_freq_path, show_plot=False)

    print("분석 완료. 이미지가 ./EDA 폴더에 저장되었습니다.")
    
    # 콤마로 구분된 값이 있는 컬럼 처리
    comma_separated_columns = ['genre_detail', 'cast', 'platform', 'production']
    results = process_multiple_comma_separated_columns(train_data, comma_separated_columns)

    # 드라마/영화 데이터 통합 분석
    analyze_drama_data(train_data, results)

    # 드라마/영화 데이터 시각화
    visualize_drama_data(train_data, results, save_path='./EDA/drama_analysis_comprehensive.png', show=False, fontprop=fontprop)

    # 장르 상세 요소 관계 시각화
    visualize_genre_relationships(results, save_path='./EDA/genre_relationship_heatmap.png', show=False, fontprop=fontprop)

    # 배우 네트워크 시각화
    visualize_actor_network(results, save_path='./EDA/actor_work_network.png', show=False, fontprop=fontprop)

    # 장르 상세 요소에 따른 평점 비교
    visualize_genre_detail_scores(results, save_path='./EDA/genre_detail_scores.png', show=False, fontprop=fontprop)

    # 플랫폼별, 장르별 작품 수 heatmap
    visualize_platform_genre_heatmap(train_data, save_path='./EDA/platform_genre_heatmap.png', fontprop=fontprop)

    # 플랫폼별, 장르 상세 요소별 작품 수 stacked bar
    visualize_platform_genre_detail_stacked_bar(train_data, save_path='./EDA/platform_genre_detail_stacked_bar.png', fontprop=fontprop, top_n=7)

    print("프로그램 종료.")