"""Covers: main = settled END card at 41.0 s without the label (label sits outside the 3:4 crop); extra = title card over the lit city."""
import importlib.util, numpy as np, cv2, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location('r', 'render.py'); r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
ti = r.text_items
r.text_items = lambda t: ti(t)[1:]  # drop the series label
cv2.imwrite('covers/cover_main.png', r.render(41.0))
r.text_items = lambda t: []
img = r.render(34.5)[:, :, 0].astype(np.float32) / 255 * 0.45
def put(text, name, size, y, col, track=-0.02):
    a, adv = r.word_img(text, name, size, track); h, w = a.shape; x = int((1080 - adv) / 2)
    img[y:y + h, x:x + w] = img[y:y + h, x:x + w] * (1 - a) + col * a
for i, l in enumerate(['GTA VI:', '475M VIEWS', 'IN A DAY']): put(l, 'InterDisplay-Bold.otf', 130, 1000 + i * 150, 1.0)
put('per Rockstar, May 2025', 'Inter-Medium.otf', 40, 1460, 0.71, 0)
put('ARCHIVE #010 · GTA VI · PART 4', 'Inter-Medium.otf', 30, 920, 0.54, 0.08)
cv2.imwrite('covers/cover_extra.png', np.repeat(np.clip(img * 255, 0, 255).astype(np.uint8)[:, :, None], 3, 2))
