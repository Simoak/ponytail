import cv2,numpy as np,sys
ts=sys.argv[1:];ims=[]
for t in ts:
    im=cv2.imread(f'stills/s_{float(t):05.2f}.png');cv2.line(im,(0,192),(1080,192),(0,0,255),3);cv2.line(im,(0,1620),(1080,1620),(0,0,255),3);cv2.line(im,(928,0),(928,1920),(0,0,255),3)
    cv2.putText(im,t,(20,1800),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,255),5);ims.append(cv2.resize(im,(360,640),interpolation=cv2.INTER_AREA))
n=5
while len(ims)%n:ims.append(np.zeros_like(ims[0]))
cv2.imwrite('stills/sheet.jpg',np.vstack([np.hstack(ims[i:i+n]) for i in range(0,len(ims),n)]))
