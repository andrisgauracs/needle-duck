from render_util import Recorder
from duck_sim import Duck
from duck_moves import perform
d = Duck(); rec = Recorder(d, 640, 480, azimuth=150, elevation=-12, distance=0.85); d.frame_cb = rec
d.settle(0.5)
seq = [("walk",{"direction":"forward","seconds":3}),("turn",{"direction":"left","degrees":180}),
 ("emote",{"expression":"look_up"}),("emote",{"expression":"look_down"}),("emote",{"expression":"look_around"}),
 ("emote",{"expression":"nod_yes"}),("shake_head",{}),("emote",{"expression":"happy"}),("emote",{"expression":"sad"}),
 ("emote",{"expression":"confused"}),("emote",{"expression":"scared"}),("emote",{"expression":"angry"}),("emote",{"expression":"sleepy"}),
 ("dance",{"style":"wiggle","seconds":4}),("dance",{"style":"headbang","seconds":3}),("dance",{"style":"spin","seconds":4}),
 ("dance",{"style":"moonwalk","seconds":4}),("dance",{"style":"chicken","seconds":4})]
marks=[]
for name,args in seq:
    rec.caption = f"{name}({', '.join(f'{k}={v!r}' for k,v in args.items())})"
    marks.append((rec.caption, len(rec.frames)))
    perform(d, {"name":name,"arguments":args})
rec.save_mp4("showreel.mp4")
print(len(rec.frames), "frames")
