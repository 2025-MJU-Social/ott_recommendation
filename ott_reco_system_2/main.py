import pandas as pd

# 1. 데이터 로드
def load_data(content_path='./data/train_data.csv', price_path='./data/ott_price.csv'):
    contents = pd.read_csv(content_path)
    prices = pd.read_csv(price_path)
    return contents, prices

# 2. 사용자 입력 받기
def get_user_input(contents):
    # 이용 가능한 base genre 목록 출력
    base_options = sorted(contents['genre'].dropna().unique())
    print('지원 가능한 장르(콤마로 다중 선택 가능):', ', '.join(base_options))
    base_genres = [g.strip() for g in input('장르 선택(영화, 드라마, 예능 등, 여러 개 가능): ').split(',')]

    # 세부 장르 옵션
    detail_options = sorted({
        x.strip() for sub in contents['genre_detail'].dropna() for x in sub.split(',')
    })
    print('세부 장르 옵션 예시(콤마로 선택 가능):', ', '.join(detail_options[:10]), '...')
    detail_genres = [g.strip() for g in input('선호 세부 장르(콤마로 구분): ').split(',')]

    age_group = input('연령대(ex: 20대, 30대): ').strip()
    gender = input('성별(male/female): ').strip()
    weekly_hours = float(input('주간 OTT 시청 시간(시간): '))
    budget = float(input('한 달 예산(원): '))

    return base_genres, detail_genres, age_group, gender, weekly_hours, budget

# 3. 러닝타임(시간) 계산
def estimate_runtime_hours(row):
    if pd.notna(row.get('runtime', None)):
        try:
            mins = int(str(row.runtime).replace('분','').strip())
            return mins / 60
        except:
            pass
    if pd.notna(row.get('episodes', None)):
        try:
            eps = int(str(row.episodes).replace('부작','').strip())
            return eps * 1.0
        except:
            pass
    return 1.0

# 4. 추천 로직 (그리디 기반 + 최소 3개, 최대 8개 추천, 중복 제거)
def recommend(contents, prices, base_genres, detail_genres, age_group, gender, weekly_hours, budget):
    max_hours = weekly_hours * 4
    desired_min, desired_max = 3, 8

    # 1차 필터: 복수 base genre, 연령대, 성별, 세부 장르 매칭
    genre_mask = contents['genre'].isin(base_genres)
    detail_mask = contents['genre_detail'].str.contains('|'.join(detail_genres), na=False)
    df = contents[genre_mask &
                  (contents.age_group == age_group) &
                  (contents.gender == gender) &
                  detail_mask].copy()

    # 필터 완화 단계
    if df.shape[0] < desired_min:
        print('⚠️ 콘텐츠 부족: 세부장르 필터 완화')
        df = contents[genre_mask &
                      (contents.age_group == age_group) &
                      (contents.gender == gender)].copy()
    if df.shape[0] < desired_min:
        print('⚠️ 콘텐츠 부족: 연령대·성별 필터만 적용')
        df = contents[genre_mask &
                      (contents.age_group == age_group)].copy()
    if df.shape[0] < desired_min:
        print('⚠️ 콘텐츠 부족: base genre 필터만 적용')
        df = contents[genre_mask].copy()
    if df.empty:
        print('⚠️ 전체 콘텐츠 중 추천')
        df = contents.copy()

    # 효율 계산 및 정렬
    df['watch_hours'] = df.apply(estimate_runtime_hours, axis=1)
    df['efficiency'] = df.score / df.watch_hours
    df = df.sort_values('efficiency', ascending=False)
    df = df.drop_duplicates(subset=['title'])

    # 그리디 선택
    selected = []
    total_hours = 0
    for _, row in df.iterrows():
        if total_hours + row.watch_hours > max_hours:
            continue
        selected.append(row)
        total_hours += row.watch_hours
        if len(selected) >= desired_max:
            break

    # 최소 갯수 부족 시 상위 효율 추천
    if len(selected) < desired_min:
        top = df.head(desired_max)
        selected = list(top.itertuples(index=False))
        total_hours = sum(r.watch_hours for r in selected)
        print(f'⚠️ 최소 {desired_min}개 부족: 효율 순 {desired_max}개 추천')

    sel_df = pd.DataFrame(selected)

    # 플랫폼 파싱
    plats = set()
    for entry in sel_df.platform.fillna('').tolist():
        for p in str(entry).split(','):
            name = p.strip()
            if name:
                plats.add(name)

    # 요금제 계산
    total_cost = 0
    plan = {}
    for p in plats:
        opts = prices[prices['서비스명'] == p]
        if opts.empty:
            continue
        cheapest = opts.loc[opts['월 구독료(원)'].idxmin()]
        plan[p] = (cheapest['요금제'], cheapest['월 구독료(원)'])
        total_cost += int(cheapest['월 구독료(원)'])

    if total_cost > budget:
        print(f"⚠️ 예산({int(budget)}원) 초과: 구독비 {total_cost}원")

    return sel_df, plan, total_hours, total_cost

# 5. 실행
if __name__ == '__main__':
    contents, prices = load_data()
    base_genres, detail_genres, age_group, gender, weekly_hours, budget = get_user_input(contents)
    sel_df, plan, hours, cost = recommend(contents, prices,
                                         base_genres, detail_genres,
                                         age_group, gender,
                                         weekly_hours, budget)

    print("\n=== 추천 구독 플랜 ===")
    for p, (pkg, c) in plan.items():
        print(f"- {p}: {pkg} / {c}원")
    print(f"총 구독비: {cost}원, 예상 시청시간: {hours:.1f}시간\n")
    print("=== 추천 콘텐츠 ===")
    for _, row in sel_df.iterrows():
        print(f"- {row.title} | {row.genre} | {row.genre_detail} | 예상 시청: {row.watch_hours:.1f}시간 | 평점: {row.score}")
