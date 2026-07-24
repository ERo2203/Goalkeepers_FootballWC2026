import pandas as pd
import numpy as np
import json
import os
import re
import html

# Load Datasets
wc = pd.read_csv('2026_World_Cup_Goalkeepers_Data.csv')
dom = pd.read_csv('Domestic_Club_Goalkeepers_Data.csv')

df = pd.merge(wc, dom, on='Player', suffixes=('_WC', '_DOM'))

# World Cup Tournament Highlights & Difficulty Context Map
wc_context = {
    'Unai Simón': ('2026 World Cup Champions 🏆', 9.5, 'Spain 🇪🇸 (FIFA #1)'),
    'Emiliano Martínez': ('Runners-up (Finalist) 🥈', 9.3, 'Argentina 🇦🇷 (FIFA #2)'),
    'Jordan Pickford': ('Semifinalists (3rd Place) 🥉', 9.1, 'England 🏴 (FIFA #4)'),
    'Mike Maignan': ('Semifinalists (4th Place)', 9.0, 'France 🇫🇷 (FIFA #3)'),
    'Alisson': ('Quarterfinalists', 8.8, 'Brazil 🇧🇷 (FIFA #5)'),
    'Yassine Bounou': ('Quarterfinalists (Highest African Rank #6)', 8.8, 'Morocco 🇲🇦 (FIFA #6)'),
    'Thibaut Courtois': ('Quarterfinalists', 8.7, 'Belgium 🇧🇪 (FIFA #8)'),
    'Gregor Kobel': ('Quarterfinalists', 8.7, 'Switzerland 🇨🇭 (FIFA #14)'),
    'Diogo Costa': ('Round of 16', 8.5, 'Portugal 🇵🇹 (FIFA #7)'),
    'Bart Verbruggen': ('Round of 16', 8.4, 'Netherlands 🇳🇱 (FIFA #9)'),
    'Guillermo Ochoa': ('Round of 16 (Co-hosts)', 8.4, 'Mexico 🇲🇽 (FIFA #10)'),
    'Camilo Vargas': ('Round of 16', 8.3, 'Colombia 🇨🇴 (FIFA #11)'),
    'Manuel Neuer': ('Round of 16', 8.3, 'Germany 🇩🇪 (FIFA #12)'),
    'Dominik Livaković': ('Round of 16', 8.2, 'Croatia 🇭🇷 (FIFA #13)'),
    'Ørjan Nyland': ('Round of 16 (Biggest Climb +12)', 8.2, 'Norway 🇳🇴 (FIFA #19)'),
    'Matt Turner': ('Round of 16 (Co-hosts)', 8.1, 'United States 🇺🇸 (FIFA #16)'),
    'Édouard Mendy': ('Round of 32', 7.9, 'Senegal 🇸🇳 (FIFA #18)'),
    'Zion Suzuki': ('Round of 32', 7.8, 'Japan 🇯🇵 (FIFA #17)'),
    'Mathew Ryan': ('Round of 32', 7.8, 'Australia 🇦🇺 (FIFA #28)'),
    'Sergio Rochet': ('Group Stage', 7.7, 'Uruguay 🇺🇾 (FIFA #20)'),
    'Fernando Muslera': ('Group Stage', 7.6, 'Uruguay 🇺🇾 (FIFA #20)'),
    'Lawrence Ati-Zigi': ('Group Stage', 7.5, 'Ghana 🇬🇭 (FIFA #65)'),
    'Alireza Beiranvand': ('Group Stage', 7.5, 'Iran 🇮🇷 (FIFA #22)'),
    'Patrick Beach': ('Round of 32', 7.4, 'Australia 🇦🇺 (FIFA #28)'),
    'Orlando Gill': ('Round of 32', 7.4, 'Paraguay 🇵🇾 (FIFA #34)'),
    'Vozinha': ('Round of 32', 7.3, 'Cabo Verde 🇨🇻 (FIFA #64)'),
    'Eloy Room': ('Group Stage', 7.2, 'Curaçao 🇨🇼 (FIFA #82)')
}

# Calculate Metrics
df['Saves_WC'] = df['Performance_Saves_WC'].fillna(0).astype(int)
df['SoTA_WC'] = df['Performance_SoTA_WC'].fillna(0).astype(int)
df['SavePct_WC'] = df['Performance_SavePct_WC'].fillna(50.0).round(1)
df['CS_WC'] = df['Performance_CS_WC'].fillna(0).astype(int)
df['GA90_WC'] = df['Performance_GA90_WC'].fillna(0.0).round(2)
df['MP_WC'] = df['Playing_Time_MP_WC'].fillna(0).astype(int)

df['Saves_DOM'] = df['Performance_Saves_DOM'].fillna(0).astype(int)
df['SoTA_DOM'] = df['Performance_SoTA_DOM'].fillna(0).astype(int)
df['SavePct_DOM'] = df['Performance_SavePct_DOM'].fillna(65.0).round(1)
df['CS_DOM'] = df['Performance_CS_DOM'].fillna(0).astype(int)
df['GA90_DOM'] = df['Performance_GA90_DOM'].fillna(0.0).round(2)
df['MP_DOM'] = df['Playing_Time_MP_DOM'].fillna(0).astype(int)

# Custom Rating Weights Model based on User Criteria:
# Priority to Number of Saves in Total Attempts (Shot Exposure) > pure Save % + Stage Depth + Domestic Comparative Score
df['SaveVolumeScore'] = np.minimum(10.0, (df['Saves_WC'] * 0.32) + (df['Saves_WC'] / np.maximum(df['SoTA_WC'], 1) * 4.5))
df['StageScore'] = df['Player'].map(lambda p: wc_context.get(p, ('Group Stage', 6.5, 'International'))[1])
df['EfficiencyScore'] = (df['SavePct_WC'] / 10.0) * 0.5 + (df['CS_WC'] * 0.6)

df['SavePctDiff'] = df['SavePct_WC'] - df['SavePct_DOM']
df['GA90Diff'] = df['GA90_DOM'] - df['GA90_WC']
df['DomesticCompScore'] = np.clip(5.0 + (df['SavePctDiff'] * 0.08) + (df['GA90Diff'] * 1.2), 3.5, 10.0)

# Final Custom Rating out of 10.0
df['Custom_Rating'] = (
    df['SaveVolumeScore'] * 0.35 +
    df['StageScore'] * 0.30 +
    df['EfficiencyScore'] * 0.20 +
    df['DomesticCompScore'] * 0.15
).round(1)

df['Custom_Rating'] = np.clip(df['Custom_Rating'], 5.2, 9.8)

# Sort strictly by Custom_Rating (Descending), then Total Saves
df = df.sort_values(by=['Custom_Rating', 'Saves_WC', 'SavePct_WC'], ascending=[False, False, False]).reset_index(drop=True)
df['Rk'] = range(1, len(df) + 1)

# Generate Comparative Domestic Story for each goalkeeper
def build_comparative_story(row):
    pname = row['Player']
    club = row['Club_WC']
    mp_wc = row['MP_WC']
    saves_wc = row['Saves_WC']
    sota_wc = row['SoTA_WC']
    sp_wc = row['SavePct_WC']
    cs_wc = row['CS_WC']
    ga90_wc = row['GA90_WC']
    
    mp_dom = row['MP_DOM']
    saves_dom = row['Saves_DOM']
    sp_dom = row['SavePct_DOM']
    cs_dom = row['CS_DOM']
    ga90_dom = row['GA90_DOM']
    
    sp_diff = round(sp_wc - sp_dom, 1)
    ga_diff = round(ga90_dom - ga90_wc, 2)
    
    clean_club = re.sub(r'^\d+\.[a-z]+\s*', '', club)
    
    if sp_diff > 0:
        sp_story = f"elevated his save efficiency by **+{sp_diff}%** on the World Cup stage (from {sp_dom}% with {clean_club} to {sp_wc}% at the WC)."
    elif sp_diff < 0:
        sp_story = f"recorded a {sp_wc}% WC save rate compared to his {sp_dom}% domestic campaign with {clean_club} across {mp_dom} league matches."
    else:
        sp_story = f"maintained consistent {sp_wc}% save efficiency across both domestic league play ({clean_club}) and the World Cup."
        
    if sota_wc > 0:
        sota_story = f"Faced **{sota_wc} shots on target** and made **{saves_wc} saves** in {mp_wc} matches under tournament pressure."
    else:
        sota_story = f"Recorded {mp_wc} World Cup appearances with {cs_wc} clean sheets."
        
    story = f"**World Cup vs. Domestic ({clean_club})**: {sota_story} In comparison to his domestic season ({saves_dom} saves in {mp_dom} matches, {cs_dom} CS), he {sp_story}"
    return story

df['Comparative_Story'] = df.apply(build_comparative_story, axis=1)

# Image Mapping
img_dir = '/Users/rohanrao/Downloads/gk data/goalkeeper_images'
players_json = []

for idx, row in df.iterrows():
    pname = str(row['Player'])
    squad = str(row['Squad_WC'])
    club = str(row['Club_WC'])
    clean_club = re.sub(r'^\d+\.[a-z]+\s*', '', club)
    
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
    avatar_path = f'assets/player-avatars/{row["Rk_WC"]}.png'
    
    highlight_tuple = wc_context.get(pname, ('World Cup Participant', 7.0, squad))
    
    players_json.append({
        'Rk': int(row['Rk']),
        'Player': pname,
        'Squad': squad,
        'Club': clean_club,
        'Age': int(row['Age_WC']) if pd.notnull(row['Age_WC']) else 'N/A',
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
        'Rating': float(row['Custom_Rating']),
        'Highlight': highlight_tuple[0],
        'FIFA_Team': highlight_tuple[2],
        'Local_File': matched_file,
        'Relative_Path': rel_path,
        'Avatar': rel_path,
        'Story': row['Comparative_Story']
    })

# Save Updated CSV and Excel Index Files
df_save = pd.DataFrame(players_json)
df_save.to_csv('Goalkeeper_Images_Index.csv', index=False)
df_save.to_excel('Goalkeeper_Images_Index.xlsx', index=False)

# Update Main CSV Datasets
df[['Rk', 'Player', 'Squad_WC', 'Club_WC', 'Custom_Rating', 'Saves_WC', 'SoTA_WC', 'SavePct_WC', 'CS_WC', 'GA90_WC', 'Comparative_Story']].to_csv('2026_World_Cup_Goalkeepers_Data.csv', index=False)
df[['Rk', 'Player', 'Squad_DOM', 'Club_DOM', 'Custom_Rating', 'Saves_DOM', 'SoTA_DOM', 'SavePct_DOM', 'CS_DOM', 'GA90_DOM', 'Comparative_Story']].to_csv('Domestic_Club_Goalkeepers_Data.csv', index=False)

# Generate Modern Single-Page Dashboard (index.html & Goalkeeper_Images_Gallery.html)
json_data = json.dumps(players_json)

html_code = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>2026 World Cup Goalkeepers Performance Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-dark: #090d16;
            --card-bg: #131b2e;
            --card-hover: #1a253e;
            --accent-blue: #38bdf8;
            --accent-indigo: #818cf8;
            --accent-gold: #fbbf24;
            --accent-emerald: #10b981;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border-color: #202c45;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        body {{
            background-color: var(--bg-dark);
            color: var(--text-main);
            padding: 2rem 1.5rem;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1450px;
            margin: 0 auto;
        }}

        /* Header & Hero Section */
        header {{
            text-align: center;
            margin-bottom: 2.5rem;
            position: relative;
        }}

        .badge-hero {{
            display: inline-block;
            background: rgba(56, 189, 248, 0.15);
            color: var(--accent-blue);
            border: 1px solid var(--accent-blue);
            padding: 0.35rem 1rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.75rem;
        }}

        header h1 {{
            font-size: 2.75rem;
            font-weight: 900;
            background: linear-gradient(135deg, #38bdf8, #818cf8, #fbbf24);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            letter-spacing: -0.02em;
        }}

        header p {{
            color: var(--text-muted);
            font-size: 1.15rem;
            max-width: 850px;
            margin: 0 auto;
            line-height: 1.6;
        }}

        /* Tournament Summary Banner */
        .tournament-summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2.5rem;
        }}

        .summary-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 1rem;
            padding: 1.25rem;
            text-align: center;
            backdrop-filter: blur(10px);
        }}

        .summary-title {{
            font-size: 0.8rem;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.05em;
            margin-bottom: 0.3rem;
        }}

        .summary-val {{
            font-size: 1.4rem;
            font-weight: 800;
            color: var(--accent-gold);
        }}

        .summary-sub {{
            font-size: 0.82rem;
            color: var(--accent-blue);
            margin-top: 0.2rem;
        }}

        /* Filter Controls */
        .controls {{
            display: flex;
            gap: 1.25rem;
            margin-bottom: 2.5rem;
            flex-wrap: wrap;
            justify-content: center;
            align-items: center;
        }}

        .search-box {{
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
        }}

        .search-box:focus {{
            border-color: var(--accent-blue);
            box-shadow: 0 0 20px rgba(56, 189, 248, 0.3);
        }}

        .sort-select {{
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
        }}

        .sort-select:focus {{
            border-color: var(--accent-indigo);
        }}

        /* Grid & Cards */
        .gallery-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 2rem;
        }}

        .card {{
            background: var(--card-bg);
            border-radius: 1.25rem;
            border: 1px solid var(--border-color);
            overflow: hidden;
            transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
            position: relative;
            display: flex;
            flex-direction: column;
        }}

        .card:hover {{
            transform: translateY(-8px);
            box-shadow: 0 20px 45px rgba(0, 0, 0, 0.65), 0 0 30px rgba(56, 189, 248, 0.25);
            border-color: var(--accent-blue);
            background: var(--card-hover);
        }}

        .card-header {{
            position: relative;
            background: linear-gradient(180deg, rgba(32, 44, 69, 0.5) 0%, rgba(9, 13, 22, 0.95) 100%);
            padding: 1.75rem 1.25rem 1rem;
            text-align: center;
        }}

        .rank-badge {{
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
        }}

        .rating-badge {{
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
        }}

        .avatar-wrapper {{
            width: 120px;
            height: 120px;
            margin: 0.6rem auto 1rem;
            border-radius: 50%;
            padding: 4px;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-indigo));
            box-shadow: 0 6px 22px rgba(0, 0, 0, 0.55);
        }}

        .avatar-img {{
            width: 100%;
            height: 100%;
            border-radius: 50%;
            object-fit: cover;
            background: #090d16;
        }}

        .player-name {{
            font-size: 1.3rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
            color: var(--text-main);
        }}

        .player-meta {{
            font-size: 0.875rem;
            color: var(--accent-blue);
            font-weight: 600;
            margin-bottom: 0.3rem;
        }}

        .tournament-highlight-pill {{
            display: inline-block;
            background: rgba(129, 140, 248, 0.15);
            color: var(--accent-indigo);
            border: 1px solid var(--accent-indigo);
            padding: 0.2rem 0.65rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            margin-top: 0.2rem;
        }}

        .card-body {{
            padding: 1.25rem;
            flex: 1;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 1.25rem;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.75rem;
            background: rgba(9, 13, 22, 0.8);
            padding: 0.9rem;
            border-radius: 0.85rem;
            border: 1px solid rgba(255, 255, 255, 0.06);
        }}

        .stat-item {{
            text-align: center;
        }}

        .stat-label {{
            font-size: 0.72rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
        }}

        .stat-value {{
            font-size: 1.15rem;
            font-weight: 800;
            color: var(--text-main);
            margin-top: 0.2rem;
        }}

        .rating-highlight {{
            color: var(--accent-gold);
        }}

        /* Comparative Domestic Performance Story Section */
        .story-section {{
            background: rgba(9, 13, 22, 0.9);
            border-radius: 0.85rem;
            border: 1px solid rgba(56, 189, 248, 0.2);
            padding: 1rem;
            font-size: 0.85rem;
            line-height: 1.55;
            color: #cbd5e1;
        }}

        .story-title {{
            font-size: 0.75rem;
            font-weight: 800;
            color: var(--accent-blue);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.4rem;
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }}

        .no-results {{
            text-align: center;
            grid-column: 1 / -1;
            padding: 4rem;
            color: var(--text-muted);
            font-size: 1.2rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <span class="badge-hero">2026 World Cup Analytics</span>
            <h1>World Cup Goalkeepers Performance Dashboard</h1>
            <p>Weighted Rating System (Priority to Save Volume under Heavy Shot Exposure & Tournament Stage) with Comparative Domestic League Analysis for all 62 World Cup Goalkeepers.</p>
        </header>

        <!-- Tournament Highlights Summary Banner -->
        <div class="tournament-summary">
            <div class="summary-card">
                <div class="summary-title">World Cup Champions</div>
                <div class="summary-val">Spain 🇪🇸</div>
                <div class="summary-sub">1–0 vs Argentina (AET)</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Runners-Up</div>
                <div class="summary-val">Argentina 🇦🇷</div>
                <div class="summary-sub">E. Martínez (20 Saves)</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">3rd Place Playoff</div>
                <div class="summary-val">England 🏴 6–4 France</div>
                <div class="summary-sub">Miami Stadium</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Highest African Rank</div>
                <div class="summary-val">Morocco 🇲🇦</div>
                <div class="summary-sub">Quarterfinalists (Y. Bounou)</div>
            </div>
        </div>

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

        function renderGallery() {{
            const filterText = document.getElementById('searchInput').value.toLowerCase();
            const sortBy = document.getElementById('sortSelect').value;
            const grid = document.getElementById('galleryGrid');
            grid.innerHTML = '';
            
            let filtered = players.filter(p => {{
                return p.Player.toLowerCase().includes(filterText) || 
                       p.Squad.toLowerCase().includes(filterText) || 
                       p.Club.toLowerCase().includes(filterText);
            }});

            if (sortBy === 'rating') {{
                filtered.sort((a, b) => b.Rating - a.Rating);
            }} else if (sortBy === 'saves') {{
                filtered.sort((a, b) => b.Saves_WC - a.Saves_WC);
            }} else if (sortBy === 'cs') {{
                filtered.sort((a, b) => b.CS_WC - a.CS_WC);
            }} else if (sortBy === 'savepct') {{
                filtered.sort((a, b) => b.SavePct_WC - a.SavePct_WC);
            }} else if (sortBy === 'name') {{
                filtered.sort((a, b) => a.Player.localeCompare(b.Player));
            }}

            if (filtered.length === 0) {{
                grid.innerHTML = '<div class="no-results">No goalkeepers match your search criteria.</div>';
                return;
            }}

            filtered.forEach((p) => {{
                const card = document.createElement('div');
                card.className = 'card';
                card.innerHTML = `
                    <div class="card-header">
                        <span class="rank-badge">#${{p.Rk}}</span>
                        <span class="rating-badge">⭐ ${{p.Rating.toFixed(1)}} / 10</span>
                        <div class="avatar-wrapper">
                            <img class="avatar-img" src="${{p.Avatar}}" alt="${{p.Player}}" onerror="this.src='https://ui-avatars.com/api/?name=${{encodeURIComponent(p.Player)}}&background=0D8ABC&color=fff&size=256'">
                        </div>
                        <h2 class="player-name">${{p.Player}}</h2>
                        <div class="player-meta">${{p.Squad}} • ${{p.Club}}</div>
                        <span class="tournament-highlight-pill">${{p.Highlight}}</span>
                    </div>
                    <div class="card-body">
                        <div class="stats-grid">
                            <div class="stat-item">
                                <div class="stat-label">Saves / Shots</div>
                                <div class="stat-value rating-highlight">${{p.Saves_WC}} / ${{p.SoTA_WC}}</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-label">Save %</div>
                                <div class="stat-value">${{p.SavePct_WC}}%</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-label">Clean Sheets</div>
                                <div class="stat-value">${{p.CS_WC}}</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-label">GA / 90</div>
                                <div class="stat-value">${{p.GA90_WC.toFixed(2)}}</div>
                            </div>
                        </div>

                        <!-- Comparative Domestic Performance Story -->
                        <div class="story-section">
                            <div class="story-title">📊 Domestic League Comparison</div>
                            <p>${{p.Story.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')}}</p>
                        </div>
                    </div>
                `;
                grid.appendChild(card);
            }});
        }}

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

print('Successfully built final dashboard (index.html & Goalkeeper_Images_Gallery.html)!')
