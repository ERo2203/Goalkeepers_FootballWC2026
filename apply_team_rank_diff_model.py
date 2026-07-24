import pandas as pd
import numpy as np
import json
import os
import re

# Load Datasets
wc = pd.read_csv('2026_World_Cup_Goalkeepers_Data.csv')
dom = pd.read_csv('Domestic_Club_Goalkeepers_Data.csv')

# Clean Country/Squad Names (Remove abbreviation prefix like 'es Spain' -> 'Spain')
def clean_country_name(name):
    if not isinstance(name, str):
        return name
    s = re.sub(r'^[a-z]{2,3}\s+', '', name)
    s = re.sub(r'^\d+\.[a-z]+\s*', '', s)
    return s.strip()

wc['Squad_WC'] = wc['Squad_WC'].apply(clean_country_name)
dom['Squad_DOM'] = dom['Squad_DOM'].apply(clean_country_name)

# Exact Match Appearances Map
wc_mp_exact = {
    'Unai Simón': 8, 'Emiliano Martínez': 8, 'Jordan Pickford': 8, 'Mike Maignan': 8,
    'Yassine Bounou': 6, 'Gregor Kobel': 6, 'Thibaut Courtois': 6, 'Ørjan Nyland': 6,
    'Alisson': 5, 'Diogo Costa': 5, 'Camilo Vargas': 5, 'Matt Turner': 5, 'Orlando Gill': 5,
    'Guillermo Ochoa': 1, 'Raúl Rangel': 4,
    'Bart Verbruggen': 4, 'Manuel Neuer': 4, 'Dominik Livaković': 4, 'Vozinha': 4,
    'Édouard Mendy': 4, 'Zion Suzuki': 4, 'Mathew Ryan': 4, 'Eloy Room': 3,
    'Lawrence Ati-Zigi': 3, 'Benjamin Asare': 2, 'Alireza Beiranvand': 3, 'Patrick Beach': 3
}

dom_mp_exact = {
    'Unai Simón': 36, 'Emiliano Martínez': 34, 'Jordan Pickford': 38, 'Mike Maignan': 35,
    'Alisson': 28, 'Thibaut Courtois': 26, 'Manuel Neuer': 23, 'Gregor Kobel': 27,
    'Yassine Bounou': 31, 'Diogo Costa': 33, 'Dominik Livaković': 34, 'Bart Verbruggen': 27,
    'Guillermo Ochoa': 22, 'Raúl Rangel': 28, 'Camilo Vargas': 32, 'Matt Turner': 18,
    'Orlando Gill': 20, 'Vozinha': 25, 'Eloy Room': 22, 'Lawrence Ati-Zigi': 30,
    'Benjamin Asare': 24, 'Alireza Beiranvand': 22, 'Patrick Beach': 24
}

# Merge Datasets
df = pd.merge(wc, dom, on='Player', suffixes=('_WC', '_DOM'))

df['MP_WC'] = df['Player'].map(lambda p: wc_mp_exact.get(p, 3))
df['MP_DOM'] = df['Player'].map(lambda p: dom_mp_exact.get(p, 25))

# FIFA Ranks
fifa_ranks = {
    'Spain': 1, 'Argentina': 2, 'France': 3, 'England': 4, 'Brazil': 5,
    'Morocco': 6, 'Portugal': 7, 'Belgium': 8, 'Netherlands': 9, 'Mexico': 10,
    'Colombia': 11, 'Germany': 12, 'Croatia': 13, 'Switzerland': 14, 'United States': 16,
    'Japan': 17, 'Senegal': 18, 'Norway': 19, 'Uruguay': 20, 'Iran': 22,
    'Austria': 23, 'Egypt': 24, 'Ecuador': 25, 'Australia': 26, 'Canada': 27,
    'Ivory Coast': 28, 'Paraguay': 34, 'South Africa': 40, 'Jordan': 45, 'DR Congo': 50,
    'Qatar': 55, 'Iraq': 60, 'Bosnia': 62, 'Saudi Arabia': 58, 'Cabo Verde': 64,
    'Ghana': 65, 'Curaçao': 82, 'Sweden': 21, 'Tunisia': 30, 'New Zealand': 70,
    'Uzbekistan': 52, 'Panama': 53, 'South Korea': 29, 'Czechia': 31, 'Haiti': 85, 'Scotland': 36
}

# Group Stage Teams Mapping
group_teams_map = {
    'Mexico': ['South Africa', 'South Korea', 'Czechia'],
    'South Africa': ['Mexico', 'South Korea', 'Czechia'],
    'South Korea': ['Mexico', 'South Africa', 'Czechia'],
    'Korea Republic': ['Mexico', 'South Africa', 'Czechia'],
    'Czechia': ['Mexico', 'South Africa', 'South Korea'],
    
    'Canada': ['Bosnia', 'Qatar', 'Switzerland'],
    'Bosnia': ['Canada', 'Qatar', 'Switzerland'],
    'Bosnia & Herzegovina': ['Canada', 'Qatar', 'Switzerland'],
    'Bosnia–Herz': ['Canada', 'Qatar', 'Switzerland'],
    'Qatar': ['Canada', 'Bosnia', 'Switzerland'],
    'Switzerland': ['Canada', 'Bosnia', 'Qatar'],
    
    'Brazil': ['Morocco', 'Haiti', 'Scotland'],
    'Morocco': ['Brazil', 'Haiti', 'Scotland'],
    'Haiti': ['Brazil', 'Morocco', 'Scotland'],
    'Scotland': ['Brazil', 'Morocco', 'Haiti'],
    
    'United States': ['Paraguay', 'Australia', 'Türkiye'],
    'USA': ['Paraguay', 'Australia', 'Türkiye'],
    'Paraguay': ['USA', 'Australia', 'Türkiye'],
    'Australia': ['USA', 'Paraguay', 'Türkiye'],
    'Türkiye': ['USA', 'Paraguay', 'Australia'],
    'Turkey': ['USA', 'Paraguay', 'Australia'],
    
    'Germany': ['Curaçao', 'Ivory Coast', 'Ecuador'],
    'Curaçao': ['Germany', 'Ivory Coast', 'Ecuador'],
    'Ivory Coast': ['Germany', 'Curaçao', 'Ecuador'],
    "Côte d'Ivoire": ['Germany', 'Curaçao', 'Ecuador'],
    'Ecuador': ['Germany', 'Curaçao', 'Ivory Coast'],
    
    'Netherlands': ['Japan', 'Sweden', 'Tunisia'],
    'Japan': ['Netherlands', 'Sweden', 'Tunisia'],
    'Sweden': ['Netherlands', 'Japan', 'Tunisia'],
    'Tunisia': ['Netherlands', 'Japan', 'Sweden'],
    
    'Belgium': ['Egypt', 'Iran', 'New Zealand'],
    'Egypt': ['Belgium', 'Iran', 'New Zealand'],
    'Iran': ['Belgium', 'Egypt', 'New Zealand'],
    'IR Iran': ['Belgium', 'Egypt', 'New Zealand'],
    'New Zealand': ['Belgium', 'Egypt', 'Iran'],
    
    'Spain': ['Cabo Verde', 'Uruguay', 'Saudi Arabia'],
    'Cabo Verde': ['Spain', 'Uruguay', 'Saudi Arabia'],
    'Uruguay': ['Spain', 'Cabo Verde', 'Saudi Arabia'],
    'Saudi Arabia': ['Spain', 'Cabo Verde', 'Uruguay'],
    
    'France': ['Senegal', 'Iraq', 'Norway'],
    'Senegal': ['France', 'Iraq', 'Norway'],
    'Iraq': ['France', 'Senegal', 'Norway'],
    'Norway': ['France', 'Senegal', 'Iraq'],
    
    'Argentina': ['Algeria', 'Austria', 'Jordan'],
    'Algeria': ['Argentina', 'Austria', 'Jordan'],
    'Austria': ['Argentina', 'Algeria', 'Jordan'],
    'Jordan': ['Argentina', 'Algeria', 'Austria'],
    
    'Portugal': ['DR Congo', 'Uzbekistan', 'Colombia'],
    'DR Congo': ['Portugal', 'Uzbekistan', 'Colombia'],
    'Congo DR': ['Portugal', 'Uzbekistan', 'Colombia'],
    'Uzbekistan': ['Portugal', 'DR Congo', 'Colombia'],
    'Colombia': ['Portugal', 'DR Congo', 'Uzbekistan'],
    
    'England': ['Croatia', 'Panama', 'Ghana'],
    'Croatia': ['England', 'Panama', 'Ghana'],
    'Panama': ['England', 'Croatia', 'Ghana'],
    'Ghana': ['England', 'Croatia', 'Panama']
}

tournament_highlights = {
    'Spain': '2026 World Cup Champions 🏆',
    'Argentina': 'Runners-up (Finalist) 🥈',
    'England': 'Semifinalists (3rd Place) 🥉',
    'France': 'Semifinalists (4th Place)',
    'Brazil': 'Quarterfinalists',
    'Morocco': 'Quarterfinalists (Highest African Rank #6)',
    'Belgium': 'Quarterfinalists',
    'Switzerland': 'Quarterfinalists',
    'Portugal': 'Round of 16',
    'Netherlands': 'Round of 16',
    'Mexico': 'Round of 16 (Co-hosts)',
    'Colombia': 'Round of 16',
    'Germany': 'Round of 16',
    'Croatia': 'Round of 16',
    'Norway': 'Round of 16 (Biggest Climb +12)',
    'United States': 'Round of 16 (Co-hosts)',
    'Paraguay': 'Round of 16',
    'Cabo Verde': 'Round of 32',
    'Senegal': 'Round of 32',
    'Japan': 'Round of 32',
    'Australia': 'Round of 32',
    'Austria': 'Round of 32',
    'Curaçao': 'Group Stage',
    'Iran': 'Group Stage',
    'IR Iran': 'Group Stage',
    'Saudi Arabia': 'Group Stage',
    'Ghana': 'Group Stage'
}

ko_opponents_map = {
    'Unai Simón': ['Uruguay', 'Germany', 'France', 'England', 'Argentina'],
    'Emiliano Martínez': ['Portugal', 'Brazil', 'Spain'],
    'Jordan Pickford': ['Switzerland', 'Spain'],
    'Mike Maignan': ['Spain'],
    'Alisson': ['Morocco'],
    'Yassine Bounou': ['Brazil'],
    'Thibaut Courtois': ['France'],
    'Gregor Kobel': ['England'],
    'Diogo Costa': ['Argentina'],
    'Bart Verbruggen': ['Spain'],
    'Manuel Neuer': ['Spain'],
    'Dominik Livaković': ['England'],
    'Ørjan Nyland': ['France'],
    'Vozinha': ['Spain']
}

# Clean Numeric Fields
df['Saves_WC'] = df['Saves_WC'].fillna(0).astype(int)
df['SoTA_WC'] = df['SoTA_WC'].fillna(0).astype(int)
df['SavePct_WC'] = df['SavePct_WC'].fillna(50.0).round(1)
df['CS_WC'] = df['CS_WC'].fillna(0).astype(int)
df['GA90_WC'] = df['GA90_WC'].fillna(0.0).round(2)

df['Saves_DOM'] = df['Saves_DOM'].fillna(0).astype(int)
df['SoTA_DOM'] = df['SoTA_DOM'].fillna(0).astype(int)
df['SavePct_DOM'] = df['SavePct_DOM'].fillna(65.0).round(1)
df['CS_DOM'] = df['CS_DOM'].fillna(0).astype(int)
df['GA90_DOM'] = df['GA90_DOM'].fillna(0.0).round(2)

# Rating Formula:
# 57% Saves Volume vs Defensive Exposure (SoTA)
# 25% Save Percentage
# 10% Clean Sheets
# 5% Opponent Rank Compared to Own Team Rank Differential (5%)
# 3% Group Stage Opponent Draw Rank (3%)
def calc_team_rank_diff_model(row):
    pname = row['Player']
    squad = clean_country_name(row['Squad_WC'])
    own_team_rank = fifa_ranks.get(squad, 30)
    
    gs_opps = group_teams_map.get(squad, ['Uruguay', 'Saudi Arabia', 'Cabo Verde'])
    ko_opps = ko_opponents_map.get(pname, [])
    
    saves = row['Saves_WC']
    sota = row['SoTA_WC']
    sp = row['SavePct_WC']
    cs = row['CS_WC']
    ga90 = row['GA90_WC'] if pd.notnull(row['GA90_WC']) else 1.5
    
    compared_ratio = (saves / np.maximum(sota, 1))
    exposure_weight = np.sqrt(sota) / 2.3
    compared_score = np.clip(compared_ratio * exposure_weight * 5.0, 3.0, 10.0)
    
    sp_score = np.clip(sp / 10.0, 4.0, 10.0)
    
    gs_ranks = [fifa_ranks.get(o, 45) for o in gs_opps]
    avg_gs_rank = np.mean(gs_ranks)
    top_gs_rank = min(gs_ranks)
    gs_rank_score = np.clip(10.5 - (top_gs_rank * 0.15 + avg_gs_rank * 0.10), 4.0, 10.0)
    
    cs_score = np.clip(4.0 + (cs * 0.85), 4.0, 10.0)
    
    if len(ko_opps) > 0:
        ko_ranks = [fifa_ranks.get(o, 30) for o in ko_opps]
        avg_opp_rank = np.mean(ko_ranks)
        rank_diff = own_team_rank - avg_opp_rank
        team_rank_diff_score = np.clip(6.5 + (rank_diff / 12.0), 4.0, 10.0)
    else:
        avg_opp_rank = np.mean(gs_ranks)
        rank_diff = own_team_rank - avg_opp_rank
        team_rank_diff_score = np.clip(6.0 + (rank_diff / 15.0), 3.0, 10.0)
        
    rating = (compared_score * 0.57) + (sp_score * 0.25) + (gs_rank_score * 0.03) + (cs_score * 0.10) + (team_rank_diff_score * 0.05)
    return round(float(np.clip(rating, 5.2, 9.8)), 1)

df['Rating_10'] = df.apply(calc_team_rank_diff_model, axis=1)

# Sort strictly by Rating_10 (Descending), then Saves_WC, then SavePct_WC
df = df.sort_values(by=['Rating_10', 'Saves_WC', 'SavePct_WC'], ascending=[False, False, False]).reset_index(drop=True)
df['Rk'] = range(1, len(df) + 1)

# Pure Stats Story
def build_pure_stats_story(row):
    club = clean_country_name(row['Club_WC'])
    mp_wc = int(row['MP_WC'])
    saves_wc = int(row['Saves_WC'])
    sota_wc = int(row['SoTA_WC'])
    sp_wc = float(row['SavePct_WC'])
    cs_wc = int(row['CS_WC'])
    
    mp_dom = int(row['MP_DOM'])
    saves_dom = int(row['Saves_DOM'])
    sp_dom = float(row['SavePct_DOM'])
    cs_dom = int(row['CS_DOM'])
    
    return f"**World Cup**: {mp_wc} MP, {sota_wc} SoTA, {saves_wc} Saves, {sp_wc}% Save%, {cs_wc} CS | **Domestic ({club})**: {mp_dom} MP, {saves_dom} Saves, {sp_dom}% Save%, {cs_dom} CS"

df['Comparative_Story'] = df.apply(build_pure_stats_story, axis=1)

# Image Mapping & JSON Export
img_dir = '/Users/rohanrao/Downloads/gk data/goalkeeper_images'
players_json = []

for idx, row in df.iterrows():
    pname = str(row['Player'])
    squad = clean_country_name(str(row['Squad_WC']))
    club = clean_country_name(str(row['Club_WC']))
    
    clean_name = re.sub(r'[^a-zA-Z0-9]', '_', pname)
    
    local_files = os.listdir(img_dir) if os.path.exists(img_dir) else []
    matched_file = None
    for f in local_files:
        if clean_name.lower() in f.lower():
            matched_file = f
            break
            
    if not matched_file:
        matched_file = f'gk_{row["Rk"]:02d}_{clean_name}.jpg'
        
    rel_path = f'goalkeeper_images/{matched_file}'
    
    real_opponents = group_teams_map.get(squad, ['Group Opponents'])
    opponents_text = ", ".join([clean_country_name(o) for o in real_opponents])
    highlight_text = tournament_highlights.get(squad, 'World Cup Participant')
    
    players_json.append({
        'Rk': int(row['Rk']),
        'Player': pname,
        'Squad': squad,
        'Club': club,
        'Age': int(row['Age_WC']) if 'Age_WC' in row and pd.notnull(row['Age_WC']) else 28,
        'MP_WC': int(row['MP_WC']),
        'Saves_WC': int(row['Saves_WC']),
        'SoTA_WC': int(row['SoTA_WC']),
        'SavePct_WC': float(row['SavePct_WC']),
        'CS_WC': int(row['CS_WC']),
        'GA90_WC': float(row['GA90_WC']),
        'MP_DOM': int(row['MP_DOM']),
        'Saves_DOM': int(row['Saves_DOM']),
        'SavePct_DOM': float(row['SavePct_DOM']),
        'CS_DOM': int(row['CS_DOM']),
        'GA90_DOM': float(row['GA90_DOM']),
        'Rating': float(row['Rating_10']),
        'Highlight': highlight_text,
        'Opponents': opponents_text,
        'Local_File': matched_file,
        'Relative_Path': rel_path,
        'Avatar': rel_path,
        'Story': row['Comparative_Story']
    })

# Save Datasets
df_save = pd.DataFrame(players_json)
df_save.to_csv('Goalkeeper_Images_Index.csv', index=False)
df_save.to_excel('Goalkeeper_Images_Index.xlsx', index=False)

df[['Rk', 'Player', 'Squad_WC', 'Club_WC', 'Rating_10', 'MP_WC', 'Saves_WC', 'SoTA_WC', 'SavePct_WC', 'CS_WC', 'GA90_WC', 'Comparative_Story']].to_csv('2026_World_Cup_Goalkeepers_Data.csv', index=False)
df[['Rk', 'Player', 'Squad_DOM', 'Club_DOM', 'Rating_10', 'MP_DOM', 'Saves_DOM', 'SoTA_DOM', 'SavePct_DOM', 'CS_DOM', 'GA90_DOM', 'Comparative_Story']].to_csv('Domestic_Club_Goalkeepers_Data.csv', index=False)

# Update HTML Dashboard (Default Borders!)
json_data = json.dumps(players_json)

html_code = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>2026 World Cup Goalkeepers Performance Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #090d16;
            --card-bg: #131b2e;
            --card-hover: #1a253e;
            --accent-blue: #38bdf8;
            --accent-indigo: #818cf8;
            --accent-gold: #fbbf24;
            --accent-silver: #e2e8f0;
            --accent-bronze: #f97316;
            --accent-emerald: #10b981;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border-color: #202c45;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        body {
            background-color: var(--bg-dark);
            color: var(--text-main);
            padding: 2.5rem 1.5rem;
            min-height: 100vh;
        }

        .container {
            max-width: 1450px;
            margin: 0 auto;
        }

        header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        header h1 {
            font-size: 2.75rem;
            font-weight: 900;
            background: linear-gradient(135deg, #38bdf8, #818cf8, #fbbf24);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            letter-spacing: -0.02em;
        }

        header p {
            color: var(--text-muted);
            font-size: 1.15rem;
            max-width: 900px;
            margin: 0 auto;
            line-height: 1.6;
        }

        .controls {
            display: flex;
            gap: 1.25rem;
            margin-bottom: 2.5rem;
            flex-wrap: wrap;
            justify-content: center;
            align-items: center;
        }

        .search-box {
            width: 100%;
            max-width: 480px;
            padding: 0.95rem 1.4rem;
            border-radius: 9999px;
            border: 1px solid var(--border-color);
            background: rgba(19, 27, 46, 0.9);
            color: white;
            font-size: 1rem;
            outline: none;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }

        .search-box:focus {
            border-color: var(--accent-blue);
            box-shadow: 0 0 20px rgba(56, 189, 248, 0.3);
        }

        .sort-select {
            padding: 0.95rem 1.4rem;
            border-radius: 9999px;
            border: 1px solid var(--border-color);
            background: rgba(19, 27, 46, 0.9);
            color: white;
            font-size: 0.95rem;
            font-weight: 600;
            outline: none;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .sort-select:focus {
            border-color: var(--accent-indigo);
        }

        .gallery-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 2rem;
        }

        .card {
            background: var(--card-bg);
            border-radius: 1.25rem;
            border: 1px solid var(--border-color);
            overflow: hidden;
            transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
            position: relative;
            display: flex;
            flex-direction: column;
        }

        .card:hover {
            transform: translateY(-8px);
            box-shadow: 0 20px 45px rgba(0, 0, 0, 0.65), 0 0 30px rgba(56, 189, 248, 0.25);
            border-color: var(--accent-blue);
            background: var(--card-hover);
        }

        /* CARD #1: GOLD COLOR WITH DEFAULT BORDER */
        .card.card-rank-1 {
            background: linear-gradient(165deg, #382c0f 0%, #261d09 50%, #151824 100%) !important;
            border: 1px solid var(--border-color) !important;
            box-shadow: 0 12px 40px rgba(251, 191, 36, 0.3) !important;
        }
        .card.card-rank-1 .card-header {
            background: linear-gradient(180deg, rgba(251, 191, 36, 0.45) 0%, rgba(56, 44, 15, 0.95) 100%) !important;
            border-bottom: 1px solid var(--border-color) !important;
        }

        /* CARD #2: SILVER COLOR WITH DEFAULT BORDER */
        .card.card-rank-2 {
            background: linear-gradient(165deg, #28364d 0%, #1c2738 50%, #131b2e 100%) !important;
            border: 1px solid var(--border-color) !important;
            box-shadow: 0 12px 40px rgba(226, 232, 240, 0.25) !important;
        }
        .card.card-rank-2 .card-header {
            background: linear-gradient(180deg, rgba(226, 232, 240, 0.4) 0%, rgba(40, 54, 77, 0.95) 100%) !important;
            border-bottom: 1px solid var(--border-color) !important;
        }

        /* CARD #3: BRONZE COLOR WITH DEFAULT BORDER */
        .card.card-rank-3 {
            background: linear-gradient(165deg, #422818 0%, #2b1a0f 50%, #131b2e 100%) !important;
            border: 1px solid var(--border-color) !important;
            box-shadow: 0 12px 40px rgba(249, 115, 22, 0.25) !important;
        }
        .card.card-rank-3 .card-header {
            background: linear-gradient(180deg, rgba(249, 115, 22, 0.4) 0%, rgba(66, 40, 24, 0.95) 100%) !important;
            border-bottom: 1px solid var(--border-color) !important;
        }

        .card-header {
            position: relative;
            background: linear-gradient(180deg, rgba(32, 44, 69, 0.5) 0%, rgba(9, 13, 22, 0.95) 100%);
            padding: 1.75rem 1.25rem 1rem;
            text-align: center;
        }

        .rank-badge {
            position: absolute;
            top: 0.85rem;
            left: 0.85rem;
            background: rgba(16, 185, 129, 0.2);
            color: var(--accent-emerald);
            border: 1px solid var(--accent-emerald);
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.82rem;
            font-weight: 800;
        }

        .rating-badge {
            position: absolute;
            top: 0.85rem;
            right: 0.85rem;
            background: rgba(251, 191, 36, 0.18);
            color: var(--accent-gold);
            border: 1px solid var(--accent-gold);
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.88rem;
            font-weight: 800;
            display: flex;
            align-items: center;
            gap: 0.3rem;
        }

        .avatar-wrapper {
            width: 120px;
            height: 120px;
            margin: 0.6rem auto 1rem;
            border-radius: 50%;
            padding: 4px;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-indigo));
            box-shadow: 0 6px 22px rgba(0, 0, 0, 0.55);
        }

        .avatar-img {
            width: 100%;
            height: 100%;
            border-radius: 50%;
            object-fit: cover;
            background: #090d16;
        }

        .player-name {
            font-size: 1.3rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
            color: var(--text-main);
        }

        .player-meta {
            font-size: 0.875rem;
            color: var(--accent-blue);
            font-weight: 600;
            margin-bottom: 0.3rem;
        }

        .tournament-highlight-pill {
            display: inline-block;
            background: rgba(129, 140, 248, 0.15);
            color: var(--accent-indigo);
            border: 1px solid var(--accent-indigo);
            padding: 0.2rem 0.65rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            margin-top: 0.2rem;
        }

        .card-body {
            padding: 1.25rem;
            flex: 1;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 1.25rem;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 0.5rem;
            background: rgba(9, 13, 22, 0.8);
            padding: 0.9rem;
            border-radius: 0.85rem;
            border: 1px solid rgba(255, 255, 255, 0.06);
        }

        .stat-item {
            text-align: center;
        }

        .stat-label {
            font-size: 0.68rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.04em;
            font-weight: 600;
        }

        .stat-value {
            font-size: 1.05rem;
            font-weight: 800;
            color: var(--text-main);
            margin-top: 0.2rem;
        }

        .rating-highlight {
            color: var(--accent-gold);
        }

        .opponents-badge {
            background: rgba(255, 255, 255, 0.04);
            border-radius: 0.65rem;
            padding: 0.65rem 0.85rem;
            font-size: 0.78rem;
            color: var(--text-muted);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .opponents-badge strong {
            color: var(--accent-blue);
        }

        .story-section {
            background: rgba(9, 13, 22, 0.9);
            border-radius: 0.85rem;
            border: 1px solid rgba(56, 189, 248, 0.2);
            padding: 1rem;
            font-size: 0.85rem;
            line-height: 1.55;
            color: #cbd5e1;
        }

        .story-title {
            font-size: 0.75rem;
            font-weight: 800;
            color: var(--accent-blue);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.4rem;
        }

        .no-results {
            text-align: center;
            grid-column: 1 / -1;
            padding: 4rem;
            color: var(--text-muted);
            font-size: 1.2rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>World Cup Goalkeepers Performance Dashboard</h1>
            <p>Weighted Rating Model (57% Saves Workload compared to Defensive Exposure SoTA, 25% Save %, 10% Clean Sheets, 5% Opponent Rank Compared to Own Team Rank, 3% Group Stage Draw Rank) for all 62 Goalkeepers.</p>
        </header>

        <div class="controls">
            <input type="text" id="searchInput" class="search-box" placeholder="Search goalkeeper, squad, or club...">
            <select id="sortSelect" class="sort-select">
                <option value="rating">Sort by Rating (High to Low)</option>
                <option value="saves">Sort by Saves Made in Total Attempts</option>
                <option value="cs">Sort by Clean Sheets</option>
                <option value="savepct">Sort by Save %</option>
                <option value="name">Sort by Name (A-Z)</option>
            </select>
        </div>

        <div class="gallery-grid" id="galleryGrid"></div>
    </div>

    <script>
        const players = {json_data};

        function renderGallery() {
            const filterText = document.getElementById('searchInput').value.toLowerCase();
            const sortBy = document.getElementById('sortSelect').value;
            const grid = document.getElementById('galleryGrid');
            grid.innerHTML = '';
            
            let filtered = players.filter(p => {
                return p.Player.toLowerCase().includes(filterText) || 
                       p.Squad.toLowerCase().includes(filterText) || 
                       p.Club.toLowerCase().includes(filterText);
            });

            if (sortBy === 'rating') {
                filtered.sort((a, b) => b.Rating - a.Rating);
            } else if (sortBy === 'saves') {
                filtered.sort((a, b) => b.Saves_WC - a.Saves_WC);
            } else if (sortBy === 'cs') {
                filtered.sort((a, b) => b.CS_WC - a.CS_WC);
            } else if (sortBy === 'savepct') {
                filtered.sort((a, b) => b.SavePct_WC - a.SavePct_WC);
            } else if (sortBy === 'name') {
                filtered.sort((a, b) => a.Player.localeCompare(b.Player));
            }

            if (filtered.length === 0) {
                grid.innerHTML = '<div class="no-results">No goalkeepers match your search criteria.</div>';
                return;
            }

            filtered.forEach((p) => {
                let cardRankClass = '';
                if (p.Rk === 1) cardRankClass = 'card-rank-1';
                else if (p.Rk === 2) cardRankClass = 'card-rank-2';
                else if (p.Rk === 3) cardRankClass = 'card-rank-3';

                const card = document.createElement('div');
                card.className = ('card ' + cardRankClass).trim();
                card.innerHTML = `
                    <div class="card-header">
                        <span class="rank-badge">#${p.Rk}</span>
                        <span class="rating-badge">⭐ ${p.Rating.toFixed(1)} / 10</span>
                        <div class="avatar-wrapper">
                            <img class="avatar-img" src="${p.Avatar}" alt="${p.Player}" onerror="this.src='https://ui-avatars.com/api/?name=${encodeURIComponent(p.Player)}&background=0D8ABC&color=fff&size=256'">
                        </div>
                        <h2 class="player-name">${p.Player}</h2>
                        <div class="player-meta">${p.Squad} • ${p.Club}</div>
                        <span class="tournament-highlight-pill">${p.Highlight}</span>
                    </div>
                    <div class="card-body">
                        <div class="stats-grid">
                            <div class="stat-item">
                                <div class="stat-label">WC Matches</div>
                                <div class="stat-value">${p.MP_WC} MP</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-label">Saves / Shots</div>
                                <div class="stat-value rating-highlight">${p.Saves_WC} / ${p.SoTA_WC}</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-label">Save %</div>
                                <div class="stat-value">${p.SavePct_WC}%</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-label">Clean Sheets</div>
                                <div class="stat-value">${p.CS_WC}</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-label">GA / 90</div>
                                <div class="stat-value">${p.GA90_WC.toFixed(2)}</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-label">League Play</div>
                                <div class="stat-value">${p.MP_DOM} MP</div>
                            </div>
                        </div>

                        <div class="opponents-badge">
                            <strong>Group Stage Draw:</strong> ${p.Opponents}
                        </div>

                        <div class="story-section">
                            <div class="story-title">Domestic League Comparison</div>
                            <p>${p.Story.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')}</p>
                        </div>
                    </div>
                `;
                grid.appendChild(card);
            });
        }

        document.getElementById('searchInput').addEventListener('input', renderGallery);
        document.getElementById('sortSelect').addEventListener('change', renderGallery);

        renderGallery();
    </script>
</body>
</html>
"""

html_code = html_code.replace('{json_data}', json_data)

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(html_code)

with open('Goalkeeper_Images_Gallery.html', 'w', encoding='utf-8') as f:
    f.write(html_code)

print('Successfully applied Opponent Rank Compared to Team Rank as 5% & Group Stage Draw as 3%!')
