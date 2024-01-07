from datetime import datetime
import math
from PIL import Image, ImageDraw
from PIL import ImageFont

WINwidth = 512
WINcolor = "#ffffff"
WINheight = WINwidth
S_length = WINwidth / 2 * 0.75
M_length = S_length * 0.95
H_length = S_length * 0.8
H_LINEwidth = 8
M_LINEwidth = int(H_LINEwidth / 2)
S_LINEwidth = 1

image = Image.new("RGB", (WINwidth, WINheight), WINcolor)
draw = ImageDraw.Draw(image)

draw.ellipse(
    [(WINwidth / 2 - 5, WINheight / 2 - 5), (WINwidth / 2 + 5, WINheight / 2 + 5)],
    fill="black",
)
draw.ellipse([(5, 5), (WINwidth - 5, WINheight - 5)], outline="black", width=2)

FontSize = int(WINwidth / 14)
Fx = 0
Fy = FontSize / 10
R = S_length + FontSize * 0.9
A = 0
for i in range(1, 13):
    A = A + 30
    Tx = R * math.cos(A / 180 * math.pi)
    Ty = R * math.sin(A / 180 * math.pi)
    draw.text(
        (WINwidth / 2 + Ty - Fx - FontSize / 2, WINheight / 2 - Tx + Fy - FontSize / 2),
        str(i),
        fill="black",
        font=ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", FontSize
        ),
    )

marking_length = S_length * 1.2
marking_width = 2
for angle in range(0, 360, 6):
    x1 = WINwidth / 2 + (marking_length - marking_width) * math.cos(
        angle / 180 * math.pi
    )
    y1 = WINheight / 2 - (marking_length - marking_width) * math.sin(
        angle / 180 * math.pi
    )
    x2 = WINwidth / 2 + marking_length * math.cos(angle / 180 * math.pi)
    y2 = WINheight / 2 - marking_length * math.sin(angle / 180 * math.pi)
    draw.line((x1, y1, x2, y2), fill="black", width=marking_width)

dial_length = S_length * 0.9
dial_width = 2
for angle in range(0, 360, 30):
    x1 = WINwidth / 2 + (dial_length - dial_width) * math.cos(angle / 180 * math.pi)
    y1 = WINheight / 2 - (dial_length - dial_width) * math.sin(angle / 180 * math.pi)
    x2 = WINwidth / 2 + dial_length * math.cos(angle / 180 * math.pi)
    y2 = WINheight / 2 - dial_length * math.sin(angle / 180 * math.pi)
    draw.line((x1, y1, x2, y2), fill="black", width=dial_width)

now = datetime.now()
nowhour = now.hour - 12 if now.hour > 12 else now.hour
nowhour = nowhour + now.minute / 60 + now.second / 3600
nowminute = now.minute + now.second / 60

H_A = nowhour / 12 * 360 * math.pi / 180
M_A = nowminute / 60 * 360 * math.pi / 180
S_A = now.second / 60 * 360 * math.pi / 180

H_x = math.cos(H_A) * H_length
H_y = math.sin(H_A) * H_length
M_x = math.cos(M_A) * M_length
M_y = math.sin(M_A) * M_length
S_x = math.cos(S_A) * S_length
S_y = math.sin(S_A) * S_length

draw.line(
    (WINwidth / 2, WINheight / 2, WINwidth / 2 + H_y, WINheight / 2 - H_x),
    fill="black",
    width=int(H_LINEwidth),
)
draw.line(
    (WINwidth / 2, WINheight / 2, WINwidth / 2 + M_y, WINheight / 2 - M_x),
    fill="black",
    width=M_LINEwidth,
)
draw.line(
    (WINwidth / 2, WINheight / 2, WINwidth / 2 + S_y, WINheight / 2 - S_x),
    fill="black",
    width=S_LINEwidth,
)

image.save("clock.png", "PNG")
