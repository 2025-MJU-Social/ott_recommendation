import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# 1. 데이터 로드
def load_data(content_path='./data/train_data.csv', price_path='./data/ott_price.csv'):
    contents = pd.read_csv(content_path)
    prices = pd.read_csv(price_path)
    return contents, prices

# 2. 사용자 입력 받기
def get_user_input(contents):
    base_options = sorted(contents['genre'].dropna().unique())
    print('지원 가능한 장르:', ', '.join(base_options))
    base_genres = [g.strip() for g in input('장르 선택(콤마 구분): ').split(',')]

    detail_opts = sorted({x.strip() for sub in contents['genre_detail'].dropna()
                          for x in sub.split(',')})
    print('세부 장르 옵션 예시:', ', '.join(detail_opts[:10]), '...')
    detail_genres = [g.strip() for g in input('선호 세부 장르(콤마 구분): ').split(',')]

    all_cast = sorted({x.strip() for sub in contents['cast'].dropna()
                       for x in sub.split(',')})
    print('출연진 옵션 예시:', ', '.join(all_cast[:10]), '...')
    preferred_cast = [c.strip() for c in input('선호 출연진(콤마 구분): ').split(',')]

    age_group = input('연령대(ex:20대): ').strip()
    gender = input('성별(male/female): ').strip()
    weekly_hours = float(input('주간 시청 시간(시간): '))
    budget = float(input('한 달 예산(원): '))

    return base_genres, detail_genres, preferred_cast, age_group, gender, weekly_hours, budget

# 3. 러닝타임 계산
def estimate_runtime_hours(row):
    if pd.notna(row.get('runtime')):
        try:
            return int(str(row.runtime).replace('분','')) / 60
        except:
            pass
    if pd.notna(row.get('episodes')):
        try:
            return int(str(row.episodes).replace('부작','')) * 1.0
        except:
            pass
    return 1.0

# 4. GCN 모델 정의
class SimpleGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# 5. 그래프 데이터 구축
def build_graph(contents, detail_genres, preferred_cast):
    items = contents['title'].tolist()
    detail_nodes = list({g.strip() for sub in contents['genre_detail'].dropna()
                         for g in sub.split(',')})
    cast_nodes = list({c.strip() for sub in contents['cast'].dropna()
                       for c in sub.split(',')})
    nodes = items + detail_nodes + cast_nodes + ['__user__']
    node_idx = {n: i for i, n in enumerate(nodes)}

    edges = []
    # item-detail edges
    for _, row in contents.iterrows():
        if pd.isna(row.genre_detail):
            continue
        i = node_idx[row.title]
        for g in str(row.genre_detail).split(','):
            g = g.strip()
            if g and g in node_idx:
                edges.append((i, node_idx[g]))
    # item-cast edges
    for _, row in contents.iterrows():
        if pd.isna(row.cast):
            continue
        i = node_idx[row.title]
        for c in str(row.cast).split(','):
            c = c.strip()
            if c and c in node_idx:
                edges.append((i, node_idx[c]))
    # user-detail & user-cast edges
    u = node_idx['__user__']
    for g in detail_genres:
        if g in node_idx:
            edges.append((u, node_idx[g]))
    for c in preferred_cast:
        if c in node_idx:
            edges.append((u, node_idx[c]))

    # undirected
    edge_index = torch.tensor(edges + [(j, i) for i, j in edges],
                              dtype=torch.long).t().contiguous()

    # one-hot features
    x = torch.eye(len(nodes), dtype=torch.float)
    return Data(x=x, edge_index=edge_index), items, node_idx

# 6. 추천
def recommend(contents, prices, base_genres, detail_genres, preferred_cast,
              age_group, gender, weekly_hours, budget):
    data, items, node_idx = build_graph(contents, detail_genres, preferred_cast)
    model = SimpleGCN(in_channels=data.num_nodes, hidden_channels=64, out_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train GCN
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        u_emb = out[node_idx['__user__']]
        loss = 0
        nbrs = data.edge_index[1][data.edge_index[0] == node_idx['__user__']]
        for nbr in nbrs:
            loss -= torch.cosine_similarity(u_emb.unsqueeze(0),
                                            out[nbr].unsqueeze(0)).mean()
        loss.backward()
        optimizer.step()

    # Inference
    model.eval()
    out = model(data.x, data.edge_index)
    u_emb = out[node_idx['__user__']]
    sims = [(title,
             torch.cosine_similarity(u_emb.unsqueeze(0),
                                     out[node_idx[title]].unsqueeze(0)).item())
            for title in items]
    sims.sort(key=lambda x: x[1], reverse=True)

    top_titles = [t for t, _ in sims[:8]]
    sel_df = contents[contents['title'].isin(top_titles)].drop_duplicates(subset=['title'])
    return sel_df, {}

# 7. 실행
if __name__ == '__main__':
    print("=== 추천 시스템 시작 ===")
    print("1. 데이터 로드")
    contents, prices = load_data()
    print("2. 사용자 입력 받기")
    base_genres, detail_genres, preferred_cast, age_group, gender, weekly_hours, budget = \
        get_user_input(contents)
    print("3. 추천 실행")
    sel_df, plan = recommend(contents, prices,
                             base_genres, detail_genres, preferred_cast,
                             age_group, gender, weekly_hours, budget)

    print("=== 추천 콘텐츠 ===")
    for _, row in sel_df.iterrows():
        print(f"- {row.title} | {row.genre_detail} | 평점: {row.score}")
