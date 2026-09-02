__version__ = '260902a'
__author__  = 'Haruka Yamaguchi'

from math import ceil
from time import time
from tkinter import Canvas, Tk

from colour import (cctf_decoding,
                    cctf_encoding,
                    Oklab_to_XYZ,
                    sRGB_to_XYZ,
                    XYZ_to_Oklab,
                    XYZ_to_sRGB)
from numpy import (array,
                   array_equal,
                   asarray,
                   average,
                   finfo,
                   hstack,
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
output_file = ''
colors = {
    (255, 205,   3): 'Yellow',
    (255, 245, 121): 'Bright Light Yellow',
    (245, 125,  32): 'Orange',
    (251, 171,  24): 'Bright Light Orange',
    (221,  26,  33): 'Red',
    (233,  93, 162): 'Dark Pink',
    (246, 173, 205): 'Bright Pink',
    (181,  28, 125): 'Magenta',
    (127,  19,  27): 'Dark Red',
    (150, 117, 180): 'Medium Lavender',
    (188, 166, 208): 'Lavender',
    ( 76,  47, 146): 'Dark Purple',
    (  0, 108, 183): 'Blue',
    ( 72, 158, 206): 'Medium Blue',
    (120, 191, 234): 'Bright Light Blue',
    (103, 130, 151): 'Sand Blue',
    (  0,  57,  94): 'Dark Blue',
    (  0, 163, 218): 'Dark Azure',
    (  0, 190, 211): 'Medium Azure',
    (204, 225, 151): 'Yellowish Green',
    (193, 228, 218): 'Light Aqua',
    (  0, 175,  77): 'Bright Green',
    (111, 148, 122): 'Sand Green',
    (  0, 146,  71): 'Green',
    (  0,  74,  45): 'Dark Green',
    (154, 202,  60): 'Lime',
    (130, 131,  83): 'Olive Green',
    (105,  46,  20): 'Reddish Brown',
    (221, 196, 142): 'Tan',
    (148, 126,  95): 'Dark Tan',
    (175, 116,  70): 'Medium Nougat',
    ( 59,  24,  13): 'Dark Brown',
    (222, 139,  95): 'Nougat',
    (252, 195, 158): 'Light Nougat',
    (166,  83,  34): 'Dark Orange',
    (244, 244, 244): 'White',
    (160, 161, 159): 'Light Bluish Gray',
    (100, 103, 101): 'Dark Bluish Gray',
    (  0,   0,   0): 'Black',
##    (230, 237, 207): 'Glow In Dark White',
##    ( 66,  66,  62): 'Pearl Dark Gray',
##    (195, 151,  55): 'Pearl Gold',
##    (135, 141, 143): 'Flat Silver'
}

def mosaic(image_in, size, colors, dither=1, pattern=None, dampen=0.5, animated=True):
    
    def rgb2lab(x):
        return XYZ_to_Oklab(sRGB_to_XYZ(x, apply_cctf_decoding=False))
    
    def lab2rgb(x):
        return XYZ_to_sRGB(Oklab_to_XYZ(x), apply_cctf_encoding=False)
    
    def fit_line(x, colors, line=range(2)):
        a, b = [colors[i] for i in line]
        ab = b - a
        ax = x - a
        d1 = ab @ ax
        if d1 <= 0:
            return a
        bx = x - b
        d2 = ab @ bx
        if d2 >= 0:
            return b
        return a + ab * d1 / (d1-d2)
    
    def fit_tri(x, colors, tri=range(3)):
        if len(colors) < 3:
            return fit_line(x, colors)
        a, b, c = [colors[i] for i in tri]
        ab = b - a
        ac = c - a
        ax = x - a
        d1 = ab @ ax
        d2 = ac @ ax
        if d1 <= 0 and d2 <= 0:
            return a
        bx = x - b
        d3 = ab @ bx
        d4 = ac @ bx
        if d3 >= 0 and d4 <= d3:
            return b
        cx = x - c
        d5 = ab @ cx
        d6 = ac @ cx
        if d6 >= 0 and d5 <= d6:
            return c
        vc = d1 * d4 - d2 * d3
        if vc <= 0 and d1 >= 0 and d3 <= 0:
            return a + ab * d1 / (d1-d3)
        vb = d2 * d5 - d1 * d6
        if vb <= 0 and d2 >= 0 and d6 <= 0:
            return a + ac * d2 / (d2-d6)
        va = d3 * d6 - d4 * d5
        if va <= 0 and d4 >= d3 and d6 <= d5:
            return b + (c-b) * (d4-d3) / (d4-d3-d6+d5)
        return a + ab * vb / (va+vb+vc) + ac * vc / (va+vb+vc)
    
    def fit_hull(x, colors, hull):
        if len(colors) < 4:
            return fit_tri(x, colors)
        test = [*x, 1] @ hull.equations.T - finfo('d').eps
        if any(test>=0):
            tris = hull.simplices[test>=0]
            if len(tris) == 1:
                return fit_tri(x, colors, tris[0])
            points = array([fit_tri(x, colors, tri) for tri in tris])
            return points[linalg.norm(x-points, axis=1).argmin()]
        else:
            return x
    
    w, h = size
    sx = ceil(image_in.size[0]/w)
    sy = ceil(image_in.size[1]/h)
    image_in = image_in.resize((w*sx, h*sy))
    image_rgb = cctf_decoding(asarray(image_in)/255)
    A = zeros((h, w, 3))
    B = zeros((h, w, 3))
    C = zeros((h+1, w+1, 3))
    D = zeros((h+1, w+1, 3))
    E = zeros((h+2, w+2, 3))
    F = zeros((h, w, 3))
    image_out = tile(array([255, 0, 255, 0], dtype='B'), (h, w, 1))
    image_out[0, 0, 3] = 255
    if len(colors) > 1:
        colors_rgb = cctf_decoding(asarray(colors)/255)
        colors_lab = rgb2lab(colors_rgb)
        tree_rgb = KDTree(colors_rgb)
        tree_lab = KDTree(colors_lab)
        hull_rgb = ConvexHull(colors_rgb) if len(colors) > 3 else None
        hull_lab = ConvexHull(colors_lab) if len(colors) > 3 else None
    if pattern is None:
        pattern = dither
    bayer = array([[[ 3/3, -3/3, -1/3], [-1/3,  3/3, -1/3]],
                   [[-1/3,  3/3, -1/3], [-1/3, -3/3,  3/3]]])
    floyd_steinberg = {(0,  1): 7/16,
                       (1, -1): 3/16,
                       (1,  0): 5/16,
                       (1,  1): 1/16}
    
    def scan0(y, x):
        if y < h and x < w:
            G = image_rgb[y*sy:(y+1)*sy, x*sx:(x+1)*sx]
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
    
    def scan1(y, x):
        if len(colors) == 0:
            return
        if len(colors) == 1:
            image_out[y, x, :3] = colors[0]
            return
        if x == 0:
            if y == 0:
                scan0(0, 0)
                scan0(0, 1)
                scan0(1, 0)
                scan0(1, 1)
            scan0(y+2, 0)
            scan0(y+2, 1)
        if y == 0:
            scan0(0, x+2)
            scan0(1, x+2)
        scan0(y+2, x+2)
        F[y, x] += E[y+1, x+1]
        if pattern > 0:
            l = E[y+1, x+1] * 9 - E[y:y+3, x:x+3].sum(axis=(0, 1))
            p = max((1-l@l/14.4)*0.05, 0)
            p = min(hstack((E[y+1, x+1], 1-E[y+1, x+1], p)))
            F[y, x] += bayer[y%2, x%2] * p * pattern
        if dither >= 1:
            i = tree_rgb.query(F[y, x])[1]
        else:
            i = tree_lab.query(rgb2lab(F[y, x]))[1]
            if dither > 0:
                i = tree_rgb.query(F[y, x] * dither + colors_rgb[i] * (1 - dither))[1]
        image_out[y, x, :3] = colors[i]
        if dampen > 0:
            F[y, x] += (fit_hull(F[y, x], colors_rgb, hull_rgb)-F[y, x]) * dampen
        e = F[y, x] - colors_rgb[i]
        for (dy, dx), a in floyd_steinberg.items():
            if y + dy < 0 or y + dy >= h or x + dx < 0 or x + dx >= w:
                continue
            F[y+dy, x+dx] += e * a * dither
    
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
        factor = min(canvas.winfo_width()/w, canvas.winfo_height()/h)
        img = image_in.resize((round(w*factor), round(h*factor)), resample=0)
        x = (canvas.winfo_width()-img.width) // 2
        y = (canvas.winfo_height()-img.height) // 2
        tkimg = ImageTk.PhotoImage(img)
        canvas.create_image(x, y, image=tkimg, anchor='nw')
        img2 = Image.fromarray(image_out).resize((img.width, img.height), resample=0)
        tkimg2 = ImageTk.PhotoImage(img2)
        id2 = canvas.create_image(x, y, image=tkimg2, anchor='nw')
        i = 0
        t = time()
    for y in range(h):
        for x in range(w):
            scan1(y, x)
            if not animated:
                continue
            if x < w - 1:
                image_out[y, x+1, 3] = 255
            elif y < h - 1:
                image_out[y+1, 0, 3] = 255
            if i > (time() - t) * 300:
                img2 = Image.fromarray(image_out).resize((img.width, img.height), resample=0)
                tkimg2 = ImageTk.PhotoImage(img2)
                canvas.itemconfig(id2, image=tkimg2)
                root.update()
            i += 1
    if animated:
        root.destroy()
    return Image.fromarray(image_out[..., :3])

if __name__ == '__main__':
    print(f'mosaic_{__version__}.py')
    input_img = Image.open(input_file)
    input_img.apply_transparency()
    input_img = input_img.convert('RGB')
    output_img = mosaic(input_img, (width, height), list(colors), dither=dither, pattern=pattern, dampen=dampen, animated=animated)
    if output_file:
        if '.' not in output_file:
            output_file += '.png'
        output_img.save(output_file)
    if result_scale > 0:
        output_img.resize((width*result_scale, height*result_scale), resample=0).show()
    for color in colors:
        i = (asarray(output_img).reshape((width*height, 3))==color).all(axis=1).sum()
        print(f'{colors[color]:<20}x{i}')
