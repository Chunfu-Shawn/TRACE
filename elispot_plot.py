import pandas as pd
import numpy as np
from plotnine import (
    ggplot, aes, geom_bar, geom_point, geom_errorbar, geom_hline,
    position_dodge, theme_bw, theme, element_text, scale_fill_manual,
    labs
)

# Raw ELISpot data (spot counts) -- each column is a condition, 4 replicates
raw = {
    'Luo-peptide-1':  [677, 684, 567, 465],
    'Luo-peptide-2':  [675, 566, 582, 443],
    'Luo-peptide-3':  [501, 322, 390, 323],
    'Luo-peptide-4':  [336, 209, 340, 220],
    'Luo-peptide-5':  [470, 311, 353, 221],
    'Luo-peptide-6':  [384, 231, 254, 276],
    'Luo-peptide-7':  [442, 237, 298, 243],
    'Luo-peptide-8':  [415, 209, 304, 204],
    'Luo-peptide-9':  [364, 206, 196, 201],
    'Luo-peptide-10': [401, 301, 276, 256],
    'CEF':            [548, 515, 593, 541],
    'CD3':            [930, 957, 813, 844],
    'DMSO':           [217, 153, 286, 289],
    'NS-peptide-HIV': [143,  77, 169, 206],
}

dmso = np.array(raw['DMSO'])
peptides = list(raw.keys())
n_reps = 4

records = []
for p in peptides:
    for i in range(n_reps):
        records.append({
            'peptide': p,
            'replicate': i + 1,
            'raw': raw[p][i],
            'norm': raw[p][i] / dmso[i],
        })

df = pd.DataFrame(records)

# Define group
def category(p):
    if p.startswith('Luo-peptide'):
        return 'Luo-peptide'
    if p in ('CEF', 'CD3'):
        return 'Positive control'
    if p == 'DMSO':
        return 'DMSO (blank)'
    return 'Negative control'

df['group'] = df['peptide'].apply(category)

# Desired x-axis order
order = (
    [f'Luo-peptide-{i}' for i in range(1, 11)]
    + ['CEF', 'CD3']
    + ['NS-peptide-HIV', 'DMSO']
)
df['peptide'] = pd.Categorical(df['peptide'], categories=order, ordered=True)

# Summary stats (groupby with observed=False preserves categorical order)
summary = (
    df.groupby('peptide', observed=False)['norm']
    .agg(mean='mean', std='std', sem=lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
    .reset_index()
)
summary['group'] = summary['peptide'].apply(category)

colors = {
    'Luo-peptide': '#4472C4',
    'Positive control': '#ED7D31',
    'Negative control': '#A5A5A5',
    'DMSO (blank)': '#FFC000',
}

p = (
    ggplot(summary, aes(x='peptide', y='mean', fill='group'))
    + geom_bar(stat='identity', width=0.7)
    + geom_errorbar(aes(ymin='mean - sem', ymax='mean + sem'), width=0.2, size=0.5)
    + geom_hline(yintercept=1, linetype='dashed', color='#999999', size=0.5)
    + geom_point(
        aes(x='peptide', y='norm', fill='group'),
        data=df,
        inherit_aes=False,
        color='black',
        stroke=0.3,
        alpha=0.8,
        size=2,
        position=position_dodge(width=0.7),
    )
    + scale_fill_manual(values=colors)
    + labs(x='Peptide', y='Normalized spot count (fold of DMSO)', fill='Group')
    + theme_bw(base_size=11)
    + theme(
        axis_text_x=element_text(angle=45, ha='right', size=9),
        figure_size=(8, 6),
        legend_position='top',
    )
)

p.save('elispot_barplot.png', dpi=150, verbose=False)
print('Done')
