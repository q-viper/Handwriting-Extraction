import argparse
import cv2
import numpy as np
import pytesseract

def tess(img, config = ('-l eng --oem 1 --psm 3')):
    # Run tesseract OCR on image
    text = pytesseract.image_to_string(img, config=config)
    return text

def crop_image(gray):
    # gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # show(img)
    bimg = cv2.GaussianBlur(gray, (5, 5), 0)
    timg = cv2.threshold(bimg, 100, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    # show(timg)
    cnts,_ = cv2.findContours(timg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts=sorted(cnts, key=cv2.contourArea)
    # print(cnts)
    segments=[]
    bbs=[]
    for cnt in cnts:
        x,y,w,h=cv2.boundingRect(cnt)
        #timg = cv2.rectangle(timg, (x, y), (x+w, y+h), 0, 3)
#         show(timg[y:y+h, x:x+w])
        segments.append(gray[y:y+h, x:x+w])
        #show(cv2.rectangle(gray, (x,y), (x+w, y+h), 100, 3))
        bbs.append([x,y,w,h])
    return segments, bbs

# Create the parser
parser = argparse.ArgumentParser(description='List the content of a folder')

# Add the arguments
parser.add_argument('-img',
                       '--img',
                       type=str,
                       help='Image path.',
                       default='F:/Desktop/projects/document scanning/Assets/handwritten/test.jpg')
parser.add_argument('-resize',
                       '--resize',
                       type=tuple,
                       help='Preprocessing Size.',
                       default=(1024, 1024))

parser.add_argument('-output',
                       '--output',
                       type=str,
                       help='Output file path.',
                       default="output.txt")






# Execute the parse_args() method
args = parser.parse_args()
# print(args)

img_name = args.img
rshape = args.resize
output= args.output

img = cv2.imread(img_name)[:]
ishape = img.shape
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray=cv2.resize(gray, (1024,1024))
gray = cv2.bilateralFilter(gray, 13, 25, 25)
gray = cv2.medianBlur(gray, 3)

bimg=cv2.GaussianBlur(gray, (7, 7), 0)
_, gimg = cv2.threshold(bimg, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

gimg = cv2.bilateralFilter(bimg, 13, 25, 25)
gimg = cv2.medianBlur(gimg, 5)

kernel=np.ones((13, 13), np.uint8)
gimg=cv2.morphologyEx(gimg, cv2.MORPH_HITMISS, kernel) # blackhat, hitmiss

#     show(gimg)
gimg = cv2.medianBlur(gimg, 3)

segs, bbs=crop_image(gimg)
txt=""
with open(output, "w") as fp:
    pass
for bb in bbs:
    x,y,w,h=bb
    simg=gray[y:y+h]
    
    # Configuration for tesseract
    config = ('-l eng --oem 3 --psm 6') # 3, 6, 1, 12

    # Run tesseract OCR on image
    text = tess(simg, config=config)

    print(text)
    
    with open(output, "a") as fp:
        fp.writelines(text)