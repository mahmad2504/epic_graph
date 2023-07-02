from requests.auth import HTTPBasicAuth
from termcolor import colored
import subprocess
import requests
import urllib3
import json
import os
from Jira import Jira
from datetime import  datetime, timedelta
import sys
import math
import platform

#import numpy as np
#import pandas as pd
#from pandas import Series, DataFrame
#import matplotlib.pyplot as plt
#from matplotlib.widgets import Slider, Button


#import plotly.express as px
#import numpy as np
#import pandas as pd



# Importing plotly.graph_objects and plotly.express
import plotly.graph_objects as plotly


if "Windows"==platform.system():
    os.system('color')

if len(sys.argv)!=2:
    print(colored("The syntax of the command is incorrect", 'red'))
    print('epic id is missing')
    exit()
    
epic=sys.argv[1]

def DrawBarChart(x,y1,y2):
    myFigure = plotly.Figure()
    myFigure.add_trace(plotly.Bar(
        x=x,
        y=y1,
        width=[0.2]*len(x),
        name="Time",
        #offset=1
    ))

    myFigure.add_trace(plotly.Bar(
        x=x,
        y=y2,
        width=[0.2]*len(x),
        offset=.2,
        name="Story Point"
    ))
     
     
     
    myFigure.update_layout(
        autosize=False,
        width=2000,
        height=700,
        yaxis=dict(
            title_text="Estimate",
            #ticktext=["little", "very long title", "long", "short title"],
            #tickvals=[2, 4, 6, 8],
            tickmode="array",
            titlefont=dict(size=20),
        )
    )
    myFigure.update_yaxes(automargin=False)
    #myFigure.show()
    print("Graph saved as graph.html")
    myFigure.write_html("graph.html")



sprint_field="customfield_11040"

fields=["summary","assignee","status","timespent","timetracking","issuelinks","description","customfield_10022"]
expand=["changelog"]

#project_id=12397

j=Jira(jiraurl="https://jira.alm.mentorg.com",jirauser="aGltcA==",jiratoken="aG1pcA==")


ids=[]
data1 = []
data2 = []
data3 = []
issues=j.Search(f" 'Epic Link' ={epic}",fields=fields,expand=[])
for issue in issues:
    
    if 'timeSpentSeconds' not in issue['fields']['timetracking']:
        issue['fields']['timetracking']['timeSpentSeconds']=0
    if issue['fields']['customfield_10022']==None:
        issue['fields']['customfield_10022']=0
        
    if issue['fields']['customfield_10022']>0 or issue['fields']['timetracking']['timeSpentSeconds']>0:
        timelog=issue['fields']['timetracking']['timeSpentSeconds']/(60*60*24)
        timelog=math.ceil(timelog)
        ids.append(issue['key'])
        data1.append(timelog)
        data2.append(issue['fields']['customfield_10022'])
        
        #print(issue['key'],issue['fields']['customfield_10022'],math.ceil(timelog))

x=["Moranp", "Turquoise", "Cornflower", "Raddles","fff"]
y1=[6, 4, 5, 11,9]
y2=[6, 4, 5, 11,9]

DrawBarChart(ids,data1,data2)
