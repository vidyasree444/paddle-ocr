from paddleocr import PaddleOCR, draw_ocr
from PIL import Image, ImageFont, ImageDraw

ocr = PaddleOCR(det_db_thresh=0.3,det_db_box_thresh=0.3,det_db_unclip_ratio=1.3,max_batch_size=10

)

img_path = 'bill.png'
results = ocr.ocr(img_path, cls=True)
print(results)
text = "\n".join([line[1][0] if line[1][0] is not None else "\n" for line in results[0]])
print(text)

image = Image.open(img_path).convert('RGB')
draw = ImageDraw.Draw(image)
font_path = "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf"
font = ImageFont.truetype(font_path, 12)


for res in results:
    for line in res:
        box = [tuple(point) for point in line[0]]
        box = [(min(point[0] for point in box), min(point[1] for point in box)),
               (max(point[0] for point in box), max(point[1] for point in box))]
        txt = line[1][0]
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0][0], box[0][1] - 25), txt, fill="blue", font=font)

image.save("result.jpg")
