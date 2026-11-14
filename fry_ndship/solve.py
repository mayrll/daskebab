import csv
import sys
from pathlib import Path
from statistics import mean
from PIL import Image, ImageDraw, ImageFont

folder=Path(__file__).resolve().parent
with Path(sys.argv[1] if len(sys.argv)>1 else folder/'potato_fields.csv').open(encoding='utf-8-sig', newline='') as f:
    rows=list(csv.DictReader(f))
long_sides=[float(row['long_side_m']) for row in rows]
p=[2*(length+10000*float(row['area_ha'])/length) for length,row in zip(long_sides,rows)]
edge=[float(row['edge_m']) for row in rows]
px,ey=mean(p),mean(edge)
slope=sum((x-px)*(y-ey) for x,y in zip(p,edge))/sum((x-px)**2 for x in p)
intercept=ey-slope*px
pred=[intercept+slope*x for x in p]
res=[y-f for y,f in zip(edge,pred)]
r2=1-sum(r*r for r in res)/sum((y-ey)**2 for y in edge)
print(f'edge = {intercept:.6f} + {slope:.8f} * P, R2={r2:.8f}')
im=Image.new('RGB',(1650,670),'white')
draw=ImageDraw.Draw(im)
font = ImageFont.load_default(size=22)
draw.text((95,20),f'Fry-ndship | Edge residuals (R² = {r2:.5f}, n = {len(rows):,})',fill='#17293c',font=font)
lo,hi=min(pred),max(pred)
for x,y in zip(pred,res):
    u=95+(x-lo)/(hi-lo)*1500
    v=60+(18-y)/36*530
    draw.ellipse((u-1.5,v-1.5,u+1.5,v+1.5),fill='#143d64')
draw.line((82,60,82,590,1605,590),fill='#17293c',width=2)
for tick in range(-15,16,5):
    v=60+(18-tick)/36*530
    draw.text((74,v),str(tick),anchor='rm',fill='#17293c',font=font)
for tick in range(300,1101,100):
    if lo<=tick<=hi:
        u=95+(tick-lo)/(hi-lo)*1500
        draw.text((u,597),str(tick),anchor='mt',fill='#17293c',font=font)
draw.text((825,635),'Fitted edge length (m)',anchor='mt',fill='#17293c',font=font)
draw.text((12,35),'Residual (m)',fill='#17293c',font=font)
im.save(folder/'residuals.png')

