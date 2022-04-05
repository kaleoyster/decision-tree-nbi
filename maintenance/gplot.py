"""
Description:
    Contains graphing functions

Author:
    Akshay Kale

Date: July 26, 2021
"""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def plot_us():
    df = px.data.election()
    geojson = px.data.election_geojson()
    fig = px.choropleth(df, geojson=geojson,
                        color="Bergeron",
                        locations="district",
                        featureidkey="properties.district",
                        projection="mercator")
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()

def df_to_plotly(df):
    """
    Convert the dataframe into dictionary
    of lists
    """
    return {'z':df.values.tolist(),
            'x':df.columns.tolist(),
            'y':df.index.tolist()}

def df_to_plotly(df):
    """
    Convert the dataframe into dictionary
    of lists
    """
    tempDict = {'High Substructure - No Deck - No Superstructure': 'Sub',
                'No Substructure - High Deck - No Superstructure': 'Deck',
                'No Substructure - No Deck - High Superstructure': 'Super'}
    newList = list()
    for series in df:
        zValues = series.values.tolist()
        zValues = [[value] for value in zValues]
        newList.append({'z':zValues,
                        'x':[series.name],
                        'y':series.index.tolist()})
    return newList

def plot_barchart1(df, name):
    """
    Plot a barchart
    x: Column name of X-axis
    y: Column name of y-axis
    """
    sort = sorted(df.items(), lambda kv:kv[2])
    print(sort[:5])
    fig = px.bar(sort,
                 x=sort.keys(),
                 y=sort.values(),
                 barmode='group')
    savefile = name + '.html'
    fig.write_html(savefile)
    fig.show()

def plot_scatterplot(df, name):
    """
    Plot a three scatter plot
    """
    fig = px.scatter_3d(df, x='supNumberIntervention',
                            y='subNumberIntervention',
                            z='deckNumberIntervention',
                            color='cluster')

    savefile = name + '.html'
    fig.write_html(savefile)
    fig.show()

def plot_barchart_sideway(df, title):
    dataHeatMap = df
    for num in range(len(df)):
        yVal = df[num].index
        xVal = list(df[num])
        name = df[num].name
        tTitle = title + ' ' + name
        fig = go.Figure(data=go.Bar(
                        x=xVal,
                        y=yVal,
                        orientation='h'))
        fig.update_layout(title_text=tTitle,
                          font_size=15,
                          yaxis=dict(
                            title='Important features',
                            titlefont_size=16,
                            tickfont_size=14),
                          xaxis=dict(
                              title='Gini Index',
                              titlefont_size=16,
                              tickfont_size=14,
                            ),
                          font=dict(size=15, color='black'),
                          plot_bgcolor='white',
                          paper_bgcolor='white')
        savefile = title + '_barchart' + '.html'
        fig.write_html(savefile)
        fig.show()

def plot_heatmap(df, title):
    dataHeatMap = df
    fig = go.Figure(data=go.Heatmap(dataHeatMap))
    fig.update_layout(title_text=title,
                      height=500,
                      width=1500,
                      font_size=15,
                      font=dict(size=15, color='black'),
                      plot_bgcolor='white',
                      paper_bgcolor='white')
    savefile = title + '.html'
    fig.write_html(savefile)
    fig.show()

def heatmap_utility(data, title, index):
    """
    plot a heatmap
    """
    fig = go.Figure(data=go.Heatmap(data,
                    colorbar=dict(title='Relevance'),
                    zmin=0,
                    zmax=0.25))
    fig.update_layout(title_text=title,
                      height=700,
                      width=500,
                      font=dict(size=13,
                                color='black'),
                      plot_bgcolor='white',
                      paper_bgcolor='white')
    savefile  = title + str(index) + '.html'
    fig.write_html(savefile)
    fig.show()

def plot_heatmap(df, title):
    """
    Create three separate heatmap
    """
    dataHeatMap = df_to_plotly(df)
    for index in range(3):
        heatmap_utility(dataHeatMap[index], title, index)

def plot_barchart(df, attribute, state, title):
    """
    Args:
        X: states
        Y: kappa or accuracy values
        names: clusters
    Returns:
        Plots
    """
    bars = list()
    savefile = title + '.html'
    clusters = df['cluster'].unique()
    for cluster in clusters:
        tempdf = df[df['cluster'].isin([cluster])]
        states = tempdf[state]
        vals =  tempdf[attribute]
        bars.append(go.Bar(name=cluster, x=states, y=vals))
    fig = go.Figure(data=bars)
    fig.update_layout(title_text=title,
                      font_size=15,
                      font=dict(size=15, color='black'),
                      xaxis=dict(title=state),
                      yaxis=dict(title=attribute),
                      plot_bgcolor='white',
                      paper_bgcolor='white',
                      barmode='group')
    fig.write_html(savefile)
    fig.show()

def plot_sankey_new(sources, targets, values, labels, title):
    """
    Description:
        Plots sankey diagram
    Args:
        sources (list)
        targets (list)
        values (list)
    Returns:
        plots
    """
    fig = go.Figure(data=[go.Sankey(
          node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),

          # Set of unique values
              label = labels,
              color = "blue"
        ),

        link = dict(
         #High Substructure
         source = sources,
         target = targets,
         value = values,

      ),
      )])
    fig.update_layout(title_text=title,
                      font_size=15,
                      font=dict(size=15, color='black'),
                      plot_bgcolor='white',
                      paper_bgcolor='white')
    fig.show()
    fig.write_html('important_features.html')
