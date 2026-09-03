__version__ = '260903a'
__author__  = 'Haruka Yamaguchi'

from math import ceil
from os import remove
from time import time
from tkinter import Canvas, Tk
from zipfile import ZipFile

from colour import (cctf_decoding,
                    Oklab_to_XYZ,
                    sRGB_to_XYZ,
                    XYZ_to_Oklab,
                    XYZ_to_sRGB)
from numpy import (array,
                   array_equal,
                   asarray,
                   average,
                   finfo,
                   linalg,
                   tile,
                   zeros)
from PIL import Image, ImageTk
from scipy.spatial import ConvexHull, KDTree

input_file   = 'C:/Users/User/Documents/Image.png'
width        = 128
height       = 96
dither       = 1.0  # 0.0 ~ 1.0
pattern      = None # 0.0 ~ 1.0, None
dampen       = 0.5  # 0.0 ~ 1.0
animated     = True
result_scale = 10
output_file  = ''
output_io    = ''
colors = {
    (  0,   0,   0): {'id':   0, 'name': 'Black'},
    (  0, 108, 183): {'id':   1, 'name': 'Blue'},
    (  0, 146,  71): {'id':   2, 'name': 'Green'},
    (221,  26,  33): {'id':   4, 'name': 'Red'},
    (233,  93, 162): {'id':   5, 'name': 'Dark Pink'},
    (  0, 175,  77): {'id':  10, 'name': 'Bright Green'},
    (255, 205,   3): {'id':  14, 'name': 'Yellow'},
    (244, 244, 244): {'id':  15, 'name': 'White'},
    (221, 196, 142): {'id':  19, 'name': 'Tan'},
    (245, 125,  32): {'id':  25, 'name': 'Orange'},
    (181,  28, 125): {'id':  26, 'name': 'Magenta'},
    (154, 202,  60): {'id':  27, 'name': 'Lime'},
    (148, 126,  95): {'id':  28, 'name': 'Dark Tan'},
    (246, 173, 205): {'id':  29, 'name': 'Bright Pink'},
    (150, 117, 180): {'id':  30, 'name': 'Medium Lavender'},
    (188, 166, 208): {'id':  31, 'name': 'Lavender'},
    (105,  46,  20): {'id':  70, 'name': 'Reddish Brown'},
    (160, 161, 159): {'id':  71, 'name': 'Light Bluish Gray'},
    (100, 103, 101): {'id':  72, 'name': 'Dark Bluish Gray'},
    ( 72, 158, 206): {'id':  73, 'name': 'Medium Blue'},
    (252, 195, 158): {'id':  78, 'name': 'Light Nougat'},
    (175, 116,  70): {'id':  84, 'name': 'Medium Nougat'},
    ( 76,  47, 146): {'id':  85, 'name': 'Dark Purple'},
    (222, 139,  95): {'id':  92, 'name': 'Nougat'},
##    ( 66,  66,  62): {'id': 148, 'name': 'Pearl Dark Gray'},
##    (135, 141, 143): {'id': 179, 'name': 'Flat Silver'},
    (251, 171,  24): {'id': 191, 'name': 'Bright Light Orange'},
    (120, 191, 234): {'id': 212, 'name': 'Bright Light Blue'},
    (255, 245, 121): {'id': 226, 'name': 'Bright Light Yellow'},
    (  0,  57,  94): {'id': 272, 'name': 'Dark Blue'},
    (  0,  74,  45): {'id': 288, 'name': 'Dark Green'},
##    (195, 151,  55): {'id': 297, 'name': 'Pearl Gold'},
    ( 59,  24,  13): {'id': 308, 'name': 'Dark Brown'},
    (127,  19,  27): {'id': 320, 'name': 'Dark Red'},
    (  0, 163, 218): {'id': 321, 'name': 'Dark Azure'},
    (  0, 190, 211): {'id': 322, 'name': 'Medium Azure'},
    (193, 228, 218): {'id': 323, 'name': 'Light Aqua'},
    (204, 225, 151): {'id': 326, 'name': 'Yellowish Green'},
##    (230, 237, 207): {'id': 329, 'name': 'Glow In Dark White'},
    (130, 131,  83): {'id': 330, 'name': 'Olive Green'},
    (111, 148, 122): {'id': 378, 'name': 'Sand Green'},
    (103, 130, 151): {'id': 379, 'name': 'Sand Blue'},
    (166,  83,  34): {'id': 484, 'name': 'Dark Orange'}
}

def mosaic(input_img, size, colors, dither=1.0,
           pattern=None, dampen=0.5, animated=False):
    
    def rgb2lab(x):
        return XYZ_to_Oklab(sRGB_to_XYZ(x, apply_cctf_decoding=False))
    
    def lab2rgb(x):
        return XYZ_to_sRGB(Oklab_to_XYZ(x), apply_cctf_encoding=False)
    
    def fit_line(x, colors, line=range(2)):
        a, b = [colors[i] for i in line]
        ab = b - a
        ax = x - a
        c = ab @ ax
        if c <= 0:
            return a
        bx = x - b
        d = ab @ bx
        if d >= 0:
            return b
        return a + ab * c / (c-d)
    
    def fit_tri(x, colors, tri=range(3)):
        if len(colors) < 3:
            return fit_line(x, colors)
        a, b, c = [colors[i] for i in tri]
        ab = b - a
        ac = c - a
        ax = x - a
        d = ab @ ax
        e = ac @ ax
        if d <= 0 and e <= 0:
            return a
        bx = x - b
        f = ab @ bx
        g = ac @ bx
        if f >= 0 and g <= f:
            return b
        cx = x - c
        h = ab @ cx
        i = ac @ cx
        if i >= 0 and h <= i:
            return c
        vc = d * g - e * f
        if vc <= 0 and d >= 0 and f <= 0:
            return a + ab * d / (d-f)
        vb = e * h - d * i
        if vb <= 0 and e >= 0 and i <= 0:
            return a + ac * e / (e-i)
        va = f * i - g * h
        if va <= 0 and g >= f and i <= h:
            return b + (c-b) * (g-f) / (g-f-i+h)
        return a + (ab*vb+ac*vc) / (va+vb+vc)
    
    def fit_hull(x, colors, hull):
        if len(colors) < 4:
            return fit_tri(x, colors)
        test = [*x, 1] @ hull.equations.T - finfo('d').eps
        if all(test<0):
            return x
        tris = hull.simplices[test>=0]
        if len(tris) == 1:
            return fit_tri(x, colors, tris[0])
        points = array([fit_tri(x, colors, tri) for tri in tris])
        return points[linalg.norm(x-points, axis=1).argmin()]
    
    def scan0(y, x):
        if y < h and x < w:
            G = input_arr[y*sy:(y+1)*sy, x*sx:(x+1)*sx]
            A[y, x] = average(G, axis=(0, 1))
            B[y, x] = average(G**2, axis=(0, 1))
            if y > 0 and x > 0:
                H = A[y-1:y+1, x-1:x+1]
                C[y, x] = average(H, axis=(0, 1))
                I = C[y, x]**2
                J = average(H**2, axis=(0, 1)) - I
                for i in range(3):
                    if J[i] >= 1e-6:
                        K = average(B[y-1:y+1, x-1:x+1, i], axis=(0, 1)) - I[i]
                        D[y, x, i] = (K/J[i])**0.5
        if y == 0 or y > h or x == 0 or x > w:
            return
        K = C[y-1:y+1, x-1:x+1]
        L = D[y-1:y+1, x-1:x+1]
        color = average(K, axis=(0, 1)) + average(L, axis=(0, 1)) * A[y-1, x-1] - average(K*L, axis=(0, 1))
        if y == 1 or y == h:
            color *= 2
        if x == 1 or x == w:
            color *= 2
        color_lab = fit_hull(rgb2lab(color.clip(0, 1)), colors_lab, hull_lab)
        E[y, x] = fit_hull(lab2rgb(color_lab), colors_rgb, hull_rgb)
        if y == 1:
            if x == 1:
                E[0, 0] = E[1, 1]
            E[0, x] = E[1, x]
            if x == w:
                E[0, w+1] = E[1, w]
        if x == 1:
            E[y, 0] = E[y, 1]
        if x == w:
            E[y, w+1] = E[y, w]
        if y == h:
            if x == 1:
                E[h+1, 0] = E[h, 1]
            E[h+1, x] = E[h, x]
            if x == w:
                E[h+1, w+1] = E[h, w]
    
    bayer = array([[[ 3/3, -3/3, -1/3], [-1/3,  3/3, -1/3]],
                   [[-1/3,  3/3, -1/3], [-1/3, -3/3,  3/3]]])
    
    floyd_steinberg = {(0,  1): 7/16,
                       (1, -1): 3/16,
                       (1,  0): 5/16,
                       (1,  1): 1/16}
    
    def scan1(y, x):
        if len(colors) == 0:
            return
        if len(colors) == 1:
            output_arr[y, x, :3] = colors[0]
            return
        for i in (y+2,) if y else range(3):
            for j in (x+2,) if x else range(3):
                scan0(i, j)
        F[y, x] += E[y+1, x+1]
        if pattern > 0:
            m = E[y+1, x+1] * 9 - E[y:y+3, x:x+3].sum(axis=(0, 1))
            n = min(max((1-m@m/14.4)*0.05, 0), *E[y+1, x+1], *(1-E[y+1, x+1]))
            F[y, x] += bayer[y%2, x%2] * n * pattern
        if dither >= 1:
            i = tree_rgb.query(F[y, x])[1]
        else:
            i = tree_lab.query(rgb2lab(F[y, x]))[1]
            if dither > 0:
                i = tree_rgb.query(F[y, x]*dither+colors_rgb[i]*(1-dither))[1]
        output_arr[y, x, :3] = colors[i]
        if dampen > 0:
            F[y, x] += (fit_hull(F[y, x], colors_rgb, hull_rgb)-F[y, x]) * dampen
        o = F[y, x] - colors_rgb[i]
        for (dy, dx), p in floyd_steinberg.items():
            if y + dy < 0 or y + dy >= h or x + dx < 0 or x + dx >= w:
                continue
            F[y+dy, x+dx] += o * p * dither
    
    input_img.apply_transparency()
    input_img = input_img.convert('RGB')
    w, h = size
    sx = ceil(input_img.size[0]/w)
    sy = ceil(input_img.size[1]/h)
    input_img = input_img.resize((w*sx, h*sy))
    if len(colors) > 1:
        colors_rgb = cctf_decoding(asarray(colors)/255)
        colors_lab = rgb2lab(colors_rgb)
        tree_rgb = KDTree(colors_rgb)
        tree_lab = KDTree(colors_lab)
        hull_rgb = ConvexHull(colors_rgb) if len(colors) > 3 else None
        hull_lab = ConvexHull(colors_lab) if len(colors) > 3 else None
    if pattern is None:
        pattern = dither
    input_arr = cctf_decoding(asarray(input_img)/255)
    A = zeros((h  , w  , 3))
    B = zeros((h  , w  , 3))
    C = zeros((h+1, w+1, 3))
    D = zeros((h+1, w+1, 3))
    E = zeros((h+2, w+2, 3))
    F = zeros((h  , w  , 3))
    output_arr = tile(array([255, 0, 255, 0], dtype='B'), (h, w, 1))
    output_arr[0, 0, 3] = 255
    if animated:
        root = Tk()
        root.state('zoomed')
        root.title('Mosaic')
        canvas = Canvas(root,
                        height=root.winfo_screenheight(),
                        highlightthickness=0,
                        width=root.winfo_screenwidth())
        canvas.pack()
        root.update()
        s = min(canvas.winfo_width()/w, canvas.winfo_height()/h)
        img0 = input_img.resize((round(w*s), round(h*s)), resample=0)
        x = (canvas.winfo_width()-img0.width) // 2
        y = (canvas.winfo_height()-img0.height) // 2
        tkimg0 = ImageTk.PhotoImage(img0)
        canvas.create_image(x, y, image=tkimg0, anchor='nw')
        img1 = Image.fromarray(output_arr).resize(img0.size, resample=0)
        tkimg1 = ImageTk.PhotoImage(img1)
        id1 = canvas.create_image(x, y, image=tkimg1, anchor='nw')
        i = 0
        t = time()
    for y in range(h):
        for x in range(w):
            scan1(y, x)
            if not animated:
                continue
            if x < w - 1:
                output_arr[y, x+1, 3] = 255
            elif y < h - 1:
                output_arr[y+1, 0, 3] = 255
            if i > (time() - t) * 300:
                img1 = Image.fromarray(output_arr).resize(img0.size, resample=0)
                tkimg1 = ImageTk.PhotoImage(img1)
                canvas.itemconfig(id1, image=tkimg1)
                root.update()
            i += 1
    if animated:
        root.destroy()
    return Image.fromarray(output_arr[..., :3])

def save_studio(filename, img, part='plate'):
    w, h = img.size
    z = h // 2 * 20 - 10
    match part:
        case 'plate':
            part_id = '3024'
    with open('model.ldr', 'wb') as f:
        for i in range(h):
            x = 10 - w // 2 * 20
            for j in range(w):
                color_id = colors[img.getpixel((j, i))]['id']
                f.write((f'1 {color_id} {x} -12 {z} 1 0 0 0 1 0 0 0 1 '
                         f'{part_id}.dat\r\n').encode())
                x += 20
            z -= 20
    img.save('thumbnail.png')
    with ZipFile(filename, 'w') as f:
        f.write('model.ldr')
        f.write('thumbnail.png')
    remove('model.ldr')
    remove('thumbnail.png')

if __name__ == '__main__':
    print(f'mosaic.py ({__version__})')
    input_img = Image.open(input_file)
    output_img = mosaic(input_img,
                        (width, height),
                        list(colors),
                        dither=dither,
                        pattern=pattern,
                        dampen=dampen,
                        animated=animated)
    if output_file:
        if '.' not in output_file:
            output_file += '.png'
        output_img.save(output_file)
    if output_io:
        if output_io[-3:] != '.io':
            output_io += '.io'
        save_studio(output_io, output_img)
    if result_scale > 0:
        output_img.resize((width*result_scale, height*result_scale), resample=0).show()
    for color in colors:
        i = (asarray(output_img).reshape((width*height, 3))==color).all(axis=1).sum()
        print(f'{colors[color]["name"]:<20}x{i}')
