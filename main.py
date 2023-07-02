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

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


import plotly.express as px
import numpy as np
import pandas as pd



# Importing plotly.graph_objects and plotly.express
import plotly.graph_objects as plotly
import plotly.express as px



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
 
    
exit()

# Sample data
df = pd.DataFrame(dict(
    group = ["A", "B", "C", "D", "E","F", "G", "H", "I", "J"],
    value = [14, 12, 8, 10, 16, 14, 12, 8, 10, 16]))

fig = px.bar(df, x = 'group', y = 'value')

fig.show()

exit()


if len(sys.argv)!=3:
    print(colored("The syntax of the command is incorrect", 'red'))
    print('(epic_report generate epic")')
    exit()
    

sprint_field="customfield_11040"
sprint_name=sys.argv[1]
sprint_id=""
board_id=sys.argv[2]
fields=["summary","assignee","status","timespent","timetracking","issuelinks","description","customfield_10022"]
expand=["changelog"]

#project_id=12397

j=Jira(jiraurl="https://jira.alm.mentorg.com",jirauser="aGltcA==",jiratoken="aG1pcA==")

ids=[]
data1 = []
data2 = []
data3 = []
issues=j.Search(" 'Epic Link' =NUC4-6808",fields=fields,expand=[])
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
        
        print(issue['key'],issue['fields']['customfield_10022'],math.ceil(timelog))
    
    
#data1 = [23,85, 72, 43, 52]
#data2 = [42, 35, 21, 16, 9]
width =.5

a=np.arange(len(data1))-width

plt.bar(a, data1, width=width,align='center')
plt.bar(np.arange(len(data2)), data2, width=width,align='center')


plt.xticks(a, ids)
plt.xticks(rotation=90)
plt.xticks(fontsize=5)

plt.text(7,7,'dd')

plt.legend(['Time Log','Story Points','ddd'])


plt.xlabel('Tasks')
plt.ylabel('Estimates')
plt.title('Epic HMIP-100')

plt.show()

exit()

print("gggg")

x=[1,2,3,4]
xn=[1.3,2.3,3.3,4.3]
y=[4,5,6,7]

plt.plot(x,y,label="line")
plt.bar(x,y,width=.3,align="center",label="bar1")
plt.bar(xn,y,width=.3,align="center",label="bar2")

plt.xlabel("X axis",fontsize=5)
plt.ylabel("Y axis")
plt.title("Title")
plt.legend()
#plt.show()

#exit()

fig,ax=plt.subplots(figsize=(10,6))

x=np.arange(1,38)
y=np.random.rand(len(x))
y1=np.random.rand(len(x))

N=20

def truncate(number, digits) -> float:
    # Improve accuracy with floating point operations, to avoid truncate(16.4, 2) = 16.39 or truncate(-1.13, 2) = -1.12
    nbDecimals = len(str(number).split('.')[1]) 
    if nbDecimals <= digits:
        return number
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper
    
def bar(pos):
    ax.set_ylabel('Scores')
    
    pos = int(pos)
    ax.clear()
    if pos+N > len(x): 
        n=len(x)-pos
    else:
        n=N
    
    X=x[pos:pos+n]
    
    Y=y[pos:pos+n]
    Y1=y1[pos:pos+n]
    ind = np.arange(n)
    ax.set_xticks(ind+0.5)
    ax.bar(X,Y,width=0.5,align='edge',color='green',ecolor='black')
    ax.bar(X+0.5,Y1,width=0.5,align='edge',color='red',ecolor='black')
    #ax.bar(X+0.5+0.5,Y1,width=0.5,align='edge',color='yellow',ecolor='black')
   
    #ax.set_xticklabels( ('2011-Jan-4', '2011-Jan-5', '2011-Jan-6') )
    for i,txt in enumerate(Y):
       #print(truncate(Y[i],1))
       ax.annotate("   "+str(truncate(Y[i],1)), (X[i],Y[i]),rotation=90)
       ax.annotate("   "+str(truncate(Y1[i],1)), (X[i]+0.5,Y1[i]),rotation=90) 
    
       
    #ax.xaxis.set_ticks([])
    #ax.yaxis.set_ticks([])

barpos = plt.axes([0.18, 0.05, 0.55, 0.03], facecolor="skyblue")
slider = Slider(barpos, 'Barpos', 0, len(x)-N, valinit=0)
slider.on_changed(bar)

bar(0)

plt.show()
exit()

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

t = np.arange(0.0, 100.0, 0.1)
s = np.sin(2*np.pi*t)
l, = plt.plot(t,s)
plt.axis([0, 10, -1, 1])

axcolor = 'lightgoldenrodyellow'
axpos = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)

spos = Slider(axpos, 'Pos', 0.1, 90.0)

def update(val):
    pos = spos.val
    ax.axis([pos,pos+10,-1,1])
    fig.canvas.draw_idle()

spos.on_changed(update)

plt.show()
exit()


N = 3
ind = np.arange(N)  # the x locations for the groups
width = 0.27       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

yvals = [4, 9, 2]
rects1 = ax.bar(ind, yvals, width, color='r')
zvals = [1,2,3]
rects2 = ax.bar(ind+width, zvals, width, color='g')
kvals = [11,12,13]
rects3 = ax.bar(ind+width*2, kvals, width, color='b')

ax.set_ylabel('Scores')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('2011-Jan-4', '2011-Jan-5', '2011-Jan-6') )
ax.legend( (rects1[0], rects2[0], rects3[0]), ('y', 'z', 'k') )

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.show()
exit()

    



