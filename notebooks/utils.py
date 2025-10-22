import numpy as np
import pandas as pd
import plotnine as gg
import re
import sys
import os
import yaml
import matplotlib.pyplot as plt
import glob 
from transformers import BertTokenizer, BertModel
import torch 
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch

def make_regret_plot(df, title, fill, metric="Cumulative Regret", bars = True,
 fill_title=None,facet_row=None, facet_col=None, include_legend=True):
    """Creates a line plot with error bars showing mean cumulative regret over time.

    Args:
        df (pandas.DataFrame): DataFrame with columns like 't', 'mean', 'ci_lb', 'ci_ub', etc.
        title (str): Title for the plot
        fill (str): Fill/color group column name
        metric (str): Y-axis label, defaults to "Cumulative Regret"
        bars (bool): Whether to include error bars
        fill_title (str): Legend title for fill
        facet_row (str): Column for faceting rows
        facet_col (str): Column for faceting columns
        include_legend (bool): Whether to include the legend

    Returns:
        plotnine.ggplot: Plot object
    """

    AGENT_COLORS = {
        'poGAMBITTS': '#0077b6',      # blue
        'foGAMBITTS': '#1f4a43',      # green
        'ens-poGAMBITTS':  '#8B0000',           # dark red
        'StdTS': '#ffd166',         # yellow
        'StdTS:Contextual' : '#ff8515' ,         # orange,
        "clarity": "#7b2cbf",              # deep purple
        "formality": "#00b4d8",            # light cyan 
        "encouragement": "#2a9d8f",        # teal 
        "optimism":  '#ff6b6b',   # light red 
        "severity": "#6c757d",             # neutral gray 
        "encouragement,clarity": "#9d4edd" ,
        "15": "#e6194B",        # bright pink-red
        "50": "#3cb44b",        # bright green
        "500": "#f58231",       # strong orange (darker than #ff8515)
        "950": "#911eb4",       # strong purple (darker than others)
        "Complete": "#46f0f0"   # neon cyan
    }
    
    # required_columns = ['t', 'mean', 'lb', 'ub', 'unique_id']
    required_columns = ["t", fill, 'ci_lb', 'ci_ub']
    if facet_row:
       required_columns += [facet_row] 

    if facet_col:
       required_columns += [facet_col] 

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"{df} is missing required columns: {missing_columns}")
    
    color_group = fill if fill else 'unique_id'
    if color_group not in df.columns:
        raise ValueError(f"'{color_group}' not in DataFrame columns.")
    
    fill_title = fill_title if fill_title and fill else "Agent"

    p = (gg.ggplot(df)
    + gg.aes('t', 'mean', colour=color_group)
    + gg.geom_line(size=1.25, alpha=0.75)
    # + gg.geom_errorbar(gg.aes(ymin='ci_lb', ymax='ci_ub', fill=color_group), alpha=0.25)
    + gg.xlab('Time Period (t)')
    + gg.ylab(f'{metric}')
    + gg.ggtitle(f'{title}')
    # + gg.scale_colour_brewer(name=fill_title, type='qual', palette='Set1')
    + gg.scale_colour_manual(name=fill_title, values=AGENT_COLORS)
    + gg.scale_fill_manual(name=fill_title, values=AGENT_COLORS)    
    + gg.theme_classic()
    # + gg.theme(text = gg.element_text(family = "Times New Roman"))
    )
    
    if bars: 
        p += gg.geom_errorbar(gg.aes(ymin='ci_lb', ymax='ci_ub', fill=color_group), alpha=0.25)      

    if not include_legend:
        p += gg.theme(legend_position='none')
    else:
        p += gg.theme(legend_position=(0.01, 0.99))  # Top-left corner inside plot

    # Add faceting if specified
    if facet_row or facet_col:
        row_str = facet_row if facet_row else '.'
        col_str = facet_col if facet_col else '.'
        facet_formula = f'{row_str} ~ {col_str}'
        p += gg.facet_grid(facet_formula)

    return p



def load_simulation_results(folder, env=None, agents=None, track_dims=None, sds=None):
    filepaths = glob.glob(os.path.join(folder, '*.csv'))

    if agents:
        filepaths = [filepath for filepath in filepaths if any(f"_{agent}_" in filepath for agent in agents)]

    if env:
        filepaths = [filepath for filepath in filepaths if os.path.basename(filepath).startswith(f"{env}_")]

    if sds:
        filepaths = [filepath for filepath in filepaths if any(f"_{sd}" in filepath for sd in sds)]

    dfs = []
    for path in filepaths:
        try:
            df = pd.read_csv(path)
            if track_dims:
                regx =re.compile(r'(' + '|'.join(track_dims) + r')')
                if regx.search(path):
                    df['dims_used'] = regx.search(path).group()
                else:
                    df['dims_used'] = df['agent_type'] 

            dfs.append(df)
        except Exception as e:
            print(f"Failed to read {path}: {e}")
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        df['agent_type'] = df['agent_type'].replace({
            'foNATS-linear': 'foGAMBITTS',
            'poNATS-linear': 'poGAMBITTS',
            'poNATS-ensemble': 'ens-poGAMBITTS',
            'MAB': 'StdTS',
            "LB": "StdTS:Linear",
            "CB": "StdTS:Contextual"
             }) 
        return df
    else:
        print("No CSV files loaded.")
        return pd.DataFrame()


def get_agg_results(df, group_ix):

    assert all(col in df.columns for col in group_ix)    
    agg_results = (df
        .assign(optimal_act = df.action == df.best_action)
        .groupby(group_ix)
        .agg({'instant_regret': ['mean', 'std', 'count'],
            "cum_regret": ['mean', 'std', 'count'],
            "optimal_act": 'mean'}))


    for metric in ['instant_regret', 'cum_regret']:
        mean = agg_results[(metric, 'mean')]
        std = agg_results[(metric, 'std')]
        count = agg_results[(metric, 'count')]
        se = std / np.sqrt(count)
        
        agg_results[(metric, 'se')] = se
        agg_results[(metric, 'ci_lb')] = mean - 1.96 * se
        agg_results[(metric, 'ci_ub')] = mean + 1.96 * se

    return agg_results

        
def make_regret_plot_matplotlib(df, title, fill='agent', metric="Cumulative Regret", bars=True,
                                 fill_title="Agent", facet_row=None, facet_col=None, include_legend=True):
    """
    Reimplementation of plotnine-style regret plot using matplotlib.
    """

    AGENT_COLORS = {
        'poGAMBITTS': '#0077b6',      # blue
        'foGAMBITTS': '#1f4a43',      # green
        'ens-poGAMBITTS':  '#8B0000',           # dark red
        'StdTS': '#ffd166',         # yellow
        'StdTS:Contextual' : '#ff8515' ,         # orange,
        "clarity": "#7b2cbf",              # deep purple
        "formality": "#00b4d8",            # light cyan 
        "encouragement": "#2a9d8f",        # teal 
        "optimism":  '#ff6b6b',   # light red 
        "severity": "#6c757d",             # neutral gray 
        "encouragement,clarity": "#9d4edd" ,
        "15 samples": "#e6194B",        # bright pink-red
        "50 samples" : "#3cb44b",        # bright green
        "100 samples" : "#2e86c1",      # medium blue
        "500 samples": "#f58231",       # strong orange (darker than #ff8515)
        "950 samples" : "#911eb4",       # strong purple (darker than others)
        "Complete": "#46f0f0"   # neon cyan
    }

    # Determine faceting grid
    row_keys = df[facet_row].unique() if facet_row else [None]
    col_keys = df[facet_col].unique() if facet_col else [None]

    n_rows, n_cols = len(row_keys), len(col_keys)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)

    for i, row_val in enumerate(row_keys):
        for j, col_val in enumerate(col_keys):
            ax = axes[i][j]

            # Filter data for this facet
            plot_df = df.copy()
            if facet_row:
                plot_df = plot_df[plot_df[facet_row] == row_val]
            if facet_col:
                plot_df = plot_df[plot_df[facet_col] == col_val]

            # Plot each agent group
            for agent, group in plot_df.groupby(fill):
                t = group['t'].values
                mean = group['mean'].values
                lb = group['ci_lb'].values
                ub = group['ci_ub'].values

                ax.plot(t, mean, label=agent, color=AGENT_COLORS.get(agent, 'gray'), linewidth=2, alpha=0.8)
                if bars:
                    ax.fill_between(t, lb, ub, color=AGENT_COLORS.get(agent, 'gray'), alpha=0.25)

            # Titles and labels
            if facet_row or facet_col:
                facet_title = ""
                if facet_row:
                    facet_title += f"{facet_row}={row_val}"
                if facet_col:
                    facet_title += f", {facet_col}={col_val}"
                ax.set_title(f"{title} ({facet_title})", loc='center')
            else:
                ax.set_title(title, loc='center')

            ax.set_xlabel("Time Period (t)")
            ax.set_ylabel(metric)

            if include_legend:
                ax.legend(title=fill_title or fill, 
                loc='center left',
                bbox_to_anchor=(1.05, 0.5),
                fontsize=14,
                title_fontsize=16,
                frameon=False,
                markerscale=2.0,
                handlelength=3.0,
                borderaxespad=1.0)
            else:
                ax.legend().set_visible(False)

    plt.tight_layout()
    return fig


## Functions to help prepare plots for submission
def modify_plot(fig, label_size=20, title_size=22,
                capitalize_legend=False, remove_title=True,
                change_legend=False):
    agents=["StdTS", "StdTS:Linear", "StdTS:Contextual", "poGAMBITTS","foGAMBITTS","ens-poGAMBITTS"]
    for ax in fig.axes:
        # Axis title and labels
        ax.title.set_fontsize(title_size)
        ax.xaxis.label.set_fontsize(label_size)
        ax.yaxis.label.set_fontsize(label_size)
        if remove_title:
            ax.title.set_text("")
        # Axis tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname("Times New Roman")
            label.set_fontsize(10)


        # Axis labels
        ax.xaxis.label.set_fontname("Times New Roman")
        ax.yaxis.label.set_fontname("Times New Roman")

        # Title
        ax.title.set_fontname("Times New Roman")

        # Legend (if any)
        legend = ax.get_legend()
        if legend:
            legend.set_title(legend.get_title().get_text(), prop={'family': 'Times New Roman'})
            for text in legend.get_texts():
                text.set_fontname("Times New Roman")
                text.set_fontsize(10)  # Legend entry text
                
                if capitalize_legend and text.get_text() not in agents:
                    text.set_text(text.get_text().capitalize())
                    
                if change_legend and text.get_text() in change_legend.keys():
                    text.set_text(change_legend[text.get_text()])
                    
                
                handles, _ = ax.get_legend_handles_labels()    
                for text, handle in zip(legend.get_texts(), handles):
                    handle.set_label(text.get_text())
                    
                legend.get_title().set_fontsize(12)
                                  
                    
                    
                    
                
            legend.get_title().set_fontsize(12) 
            
        
    # Optional: Adjust layout
    fig.tight_layout()
    
    return fig

def save_fig(fig, fig_name, fig_dir):
    fig.savefig(os.path.join(fig_dir, fig_name), dpi=300, bbox_inches='tight')
    plt.close(fig)




class VAEDataset(Dataset):
    def __init__(self, data_folder= "data/simulations/", file="optimism_interventions_5000.csv", 
                 text_column="text"):
        csv_file = os.path.join(data_folder, file)
        self.data = pd.read_csv(csv_file)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        self.text_column = text_column
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.model.eval()
        self.text_embeddings = self.preprocess_all_texts(batch_size=128)
        self.features = self.text_embeddings 


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.features[idx]
        label = self.labels.iloc[idx].to_dict()


        return sample, label

    @staticmethod
    def collate_fn(batch):
        # list of tuples
        # for each tuple, first element is a feature tensor, second element is a dictionary
        features, labels = zip(*batch)
        return features, labels

    @torch.no_grad()
    def transform_texts_in_batch(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding="longest", truncation=True, max_length=512)
        for k, v in inputs.items():
            inputs[k] = v.cuda()
        with torch.no_grad():
            outputs = self.model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return cls_embeddings.cpu()

    def preprocess_all_texts(self, batch_size=16):
        text_column = self.data[self.text_column].tolist()
        all_embeddings = []
        
        # Process texts in batches
        for i in tqdm(range(0, len(text_column), batch_size)):
            batch_texts = text_column[i:i + batch_size]
            batch_embeddings = self.transform_texts_in_batch(batch_texts)
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all batch embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings